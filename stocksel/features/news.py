"""News sentiment: article-level scoring, timestamp alignment, LLM cutoff guard, decay tracking.

Key rules
* Article-level: the scorer reads headline + full body (headline-only is weaker).
* Timestamp alignment: an article is first usable at the first close strictly
  after its timestamp (market time zone). Signals at the close, trades at the next open.
* LLM knowledge cutoff: an LLM that was trained on text from the evaluation
  period can "know" what happened next. Articles at or before the scorer's
  ``knowledge_cutoff`` are dropped, so the news signal only exists after it.
  Unknown cutoff => treated as today => no historical news signal at all.
* Decay: the stand-alone daily news strategy is tracked with a rolling Sharpe
  (net of a harsher cost model); its weight = clip(rolling SR / target, 0, 1)
  scales the news feature automatically as the edge fades.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..data.bundle import DataBundle
from ..data.pit import decision_timestamps
from ..errors import LeakageError

# Compact finance lexicon in the spirit of Loughran & McDonald (2011).
POSITIVE = {"beat", "beats", "strong", "growth", "record", "upgrade", "upgraded", "surge", "profit", "profitable",
            "exceeded", "exceeds", "robust", "outperform", "expansion", "gain", "gains", "improved", "improve",
            "momentum", "raised", "raises", "breakthrough", "win", "wins", "accelerating", "favorable",
            "optimistic", "rebound", "strength", "upside", "approval", "approved"}
NEGATIVE = {"miss", "misses", "weak", "weakness", "decline", "declines", "loss", "losses", "downgrade",
            "downgraded", "plunge", "lawsuit", "shortfall", "impairment", "recall", "investigation", "default",
            "cut", "cuts", "warning", "warns", "slump", "layoffs", "fraud", "deteriorating", "adverse",
            "bankruptcy", "restatement", "delay", "probe", "penalty", "writedown", "halt"}
_TOKEN = re.compile(r"[a-z]+")


class SentimentScorer:
    name = "base"
    knowledge_cutoff: Optional[pd.Timestamp] = None

    def score(self, articles: pd.DataFrame) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class LexiconScorer(SentimentScorer):
    """Deterministic dictionary scorer: no training data, so no look-ahead contamination."""
    name = "lexicon"

    def score(self, articles: pd.DataFrame) -> np.ndarray:
        text = (articles["headline"].fillna("") + " " + articles["body"].fillna("")).str.lower()
        out = np.empty(len(text))
        for k, t in enumerate(text.to_numpy()):
            toks = _TOKEN.findall(t)
            p = sum(w in POSITIVE for w in toks)
            n = sum(w in NEGATIVE for w in toks)
            out[k] = (p - n) / (p + n + 1.0)
        return out


class ClaudeScorer(SentimentScorer):
    """Article-level sentiment with Claude (structured JSON output, on-disk cache).

    Set ``news.llm.knowledge_cutoff`` to the LATEST training cutoff among the
    primary model and any fallback model that may serve the request. Until it is
    set, the cutoff is assumed to be today (no historical evaluation allowed).
    """
    name = "claude"

    SCHEMA = {
        "type": "object",
        "properties": {
            "sentiment": {"type": "number", "description": "-1 very negative ... +1 very positive for the company"},
            "relevance": {"type": "number", "description": "0 unrelated ... 1 central to the company"},
        },
        "required": ["sentiment", "relevance"],
        "additionalProperties": False,
    }

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low", knowledge_cutoff=None,
                 cache_file: str = "artifacts/cache/news_scores.jsonl", max_body_chars: int = 12000):
        self.model = model
        self.effort = effort
        self.knowledge_cutoff = pd.Timestamp(knowledge_cutoff) if knowledge_cutoff else pd.Timestamp.today().normalize()
        self.cutoff_assumed = knowledge_cutoff is None
        self.cache_path = Path(cache_file)
        self.max_body_chars = max_body_chars
        self._cache = self._load_cache()
        self._client = None

    def _load_cache(self) -> dict:
        cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                rec = json.loads(line)
                cache[rec["key"]] = rec["score"]
        return cache

    def _key(self, ticker, headline, body) -> str:
        return hashlib.sha256(f"{self.model}|{ticker}|{headline}|{body}".encode()).hexdigest()

    def _client_(self):
        if self._client is None:
            import anthropic  # optional dependency
            self._client = anthropic.Anthropic()
        return self._client

    def _score_one(self, ticker: str, headline: str, body: str) -> float:
        import anthropic
        body = body[: self.max_body_chars]  # long bodies: cap is configurable; raise it rather than lose content
        prompt = (f"Company ticker: {ticker}\nHeadline: {headline}\n\nFull article:\n{body}\n\n"
                  "Rate the tone of this article toward the company's business prospects. "
                  "Use only the article text; do not use any knowledge of later events or prices.")
        try:
            resp = self._client_().beta.messages.create(
                model=self.model,
                max_tokens=1024,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": self.SCHEMA}},
                system="You are a careful financial news analyst producing calibrated sentiment scores.",
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.RateLimitError:
            raise
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise
            return float("nan")
        if resp.stop_reason == "refusal":
            return float("nan")
        text = next((b.text for b in resp.content if b.type == "text"), None)
        if text is None:
            return float("nan")
        data = json.loads(text)
        return float(np.clip(data["sentiment"], -1, 1) * np.clip(data["relevance"], 0, 1))

    def score(self, articles: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(articles))
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a") as fh:
            for k, row in enumerate(articles.itertuples(index=False)):
                key = self._key(row.ticker, row.headline, row.body)
                if key not in self._cache:
                    s = self._score_one(row.ticker, row.headline or "", row.body or "")
                    self._cache[key] = s
                    fh.write(json.dumps({"key": key, "score": s}) + "\n")
                out[k] = self._cache[key]
        return out


class PrecomputedScorer(SentimentScorer):
    """Scores from a file: ticker,timestamp,score (and the cutoff of the model that produced them)."""
    name = "precomputed"

    def __init__(self, path: str, knowledge_cutoff=None):
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        self.table = df.set_index(["ticker", "timestamp"])["score"]
        self.knowledge_cutoff = pd.Timestamp(knowledge_cutoff) if knowledge_cutoff else pd.Timestamp.today().normalize()

    def score(self, articles: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays([articles["ticker"], pd.to_datetime(articles["timestamp"], utc=True)])
        return self.table.reindex(idx).to_numpy(dtype=float)


def make_scorer(cfg) -> SentimentScorer:
    n = cfg.news
    if n.scorer == "lexicon":
        return LexiconScorer()
    if n.scorer == "claude":
        return ClaudeScorer(model=n.llm.model, effort=n.llm.effort, knowledge_cutoff=n.llm.knowledge_cutoff,
                            cache_file=n.llm.cache_file, max_body_chars=n.llm.max_body_chars)
    if n.scorer == "precomputed":
        return PrecomputedScorer(n.precomputed_file, n.llm.knowledge_cutoff)
    raise ValueError(n.scorer)


def assert_cutoff_before(scorer: SentimentScorer, eval_start) -> None:
    c = scorer.knowledge_cutoff
    if c is not None and pd.Timestamp(c) >= pd.Timestamp(eval_start):
        raise LeakageError(f"{scorer.name}: knowledge cutoff {pd.Timestamp(c).date()} is not before the evaluation "
                           f"start {pd.Timestamp(eval_start).date()} — the model may know the outcomes.")


def effective_day_positions(ts_utc: pd.Series, dates: pd.DatetimeIndex, tz: str, close_time: str) -> np.ndarray:
    """Index of the first trading day whose close is strictly after the article timestamp."""
    local = pd.to_datetime(ts_utc, utc=True).dt.tz_convert(tz).dt.tz_localize(None)
    closes = decision_timestamps(dates, close_time).as_unit("ns")
    return closes.searchsorted(pd.DatetimeIndex(local).as_unit("ns"), side="right")


def daily_sentiment(bundle: DataBundle, scorer: SentimentScorer, cfg) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Sum and count of article scores per (effective trading day, ticker)."""
    dates, tick = bundle.dates, bundle.tickers
    z = pd.DataFrame(0.0, index=dates, columns=tick)
    info = {"n_articles": 0, "n_dropped_cutoff": 0, "scorer": scorer.name,
            "cutoff": None if scorer.knowledge_cutoff is None else str(pd.Timestamp(scorer.knowledge_cutoff).date())}
    if bundle.news is None or len(bundle.news) == 0:
        return z, z.copy(), info
    art = bundle.news[bundle.news["ticker"].isin(tick)].copy()
    info["n_articles"] = len(art)
    if scorer.knowledge_cutoff is not None:
        cut = pd.Timestamp(scorer.knowledge_cutoff).tz_localize("UTC") + pd.Timedelta(days=1)
        keep = pd.to_datetime(art["timestamp"], utc=True) >= cut
        info["n_dropped_cutoff"] = int((~keep).sum())
        art = art[keep]
    if len(art) == 0:
        return z, z.copy(), info
    pos = effective_day_positions(art["timestamp"], dates, cfg.market.timezone, cfg.market.close_time)
    ok = pos < len(dates)
    art, pos = art[ok], pos[ok]
    scores = scorer.score(art)
    good = np.isfinite(scores)
    col = tick.get_indexer(art["ticker"])
    s = np.zeros((len(dates), len(tick)))
    c = np.zeros((len(dates), len(tick)))
    np.add.at(s, (pos[good], col[good]), scores[good])
    np.add.at(c, (pos[good], col[good]), 1.0)
    return pd.DataFrame(s, index=dates, columns=tick), pd.DataFrame(c, index=dates, columns=tick), info


def monthly_news_feature(ssum: pd.DataFrame, scnt: pd.DataFrame, dates: pd.DatetimeIndex, lookback: int) -> pd.DataFrame:
    s = ssum.rolling(lookback, min_periods=1).sum()
    c = scnt.rolling(lookback, min_periods=1).sum()
    return (s / c.replace(0, np.nan)).loc[dates]


def news_weight(net_daily: pd.Series, window: int, target_sharpe: float, min_obs: int) -> pd.Series:
    """Automatic down-weighting as the news edge decays. Uses only realized past returns."""
    r = net_daily.shift(1)  # interval t is realized at the next open
    mu = r.rolling(window, min_periods=min_obs).mean()
    sd = r.rolling(window, min_periods=min_obs).std()
    sr = mu / sd * np.sqrt(252)
    return (sr / target_sharpe).clip(0, 1).fillna(0.0).rename("news_weight")
