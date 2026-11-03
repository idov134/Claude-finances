# stocksel — מערכת מחקר ספקנית לבחירת מניות

## GPT branch: research dashboard and execution audit

This version builds on Claude's `ccr-34655f6e-o332cd` branch. It adds a local,
responsive research dashboard, asynchronous quote feeds, stricter validation
promotion, and fixes execution/return-accounting defects. **It has not established
a real-market trading edge.** The one tested strategy change remains off by default;
see [the algorithm review and reproducible evidence](docs/ALGO_REVIEW.md).

### Start the app

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m stocksel app
```

The server listens locally on port 8000. The first scan loads the configured
history and fits the baseline in a worker thread; the UI remains responsive.
The default **synthetic replay works offline without API keys**. It excludes the
previously used final 24-month holdout, scores every eligible ticker using the
existing size/liquidity filters, and reserves a subsequent historical month for
replay. Replayed event timestamps remain historical and never display as live.
With no frozen `final_spec.json`, the baseline is explicitly provisional and
has no performance claims; no research or holdout is opened automatically.

The five screens cover freshness/coverage, sortable stock research and details,
the existing paper ledger, validation/decay, and risk alerts. Every quote shows
event time, receipt time, age and assumed provider delay; recommendations show
the separate signal date and age. Light/dark themes and phone widths are supported.
The paper ledger starts in cash and is read-only in the app: submit/update paper
orders explicitly with the existing CLI. No brokerage execution is connected.

FastAPI plus a plain JavaScript frontend keeps the UI asynchronous without a second
Node build/runtime or external CDN. Quotes poll separately from the hourly
new-completed-month check. Historical data refreshes daily for real providers;
monthly model fitting is skipped when the decision month has not changed.
News trading remains disabled because the documented net-cost validation failed;
there is no connected real-time news/event strategy.

### Real quote feeds

Configure a **real historical provider and eligible universe** in `config.yaml`
first (prefer point-in-time, survivorship-free vendor exports or Sharadar).
Synthetic `S0000` tickers cannot be combined with exchange feeds. A frozen spec
must match the configuration; stale/mismatched specs fail visibly.

```bash
python -m stocksel app --feed alpaca
# Or delayed polling:
python -m pip install 'yfinance>=0.2'
python -m stocksel app --feed yfinance
```

Alpaca reads `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` from the environment only;
optional `ALPACA_DATA_FEED=iex` (default) or `sip` selects the entitlement.
IEX is one exchange, **not a consolidated whole-market tape**. SIP requires the
appropriate subscription. Yahoo is conservatively classified as delayed by at
least 900 seconds; actual timestamps still determine staleness. Missing symbols,
disconnections and future timestamps cannot turn the global indicator green.
US exchange holidays, daylight saving and early closes use an exchange calendar.
Websocket reconnects and polling backoff retain the original quote timestamps.

Real credentials/provider entitlements were unavailable for end-to-end vendor
verification. Historical adapters remain subject to the limitations below.
Raw real-time prices are **not yet reconciled with adjusted historical shares
and corporate actions**, so the app withholds real-feed stop comparisons and
quote-based ledger marks; it retains and labels the ledger's historical marks.
Synthetic holdings support fixed and ATR trailing-stop alerts, using only bars
before each replay event. No current prices or P&L are invented to fill gaps.

Only bind beyond `127.0.0.1` behind authenticated access; the local app has no
user authentication. Deploying it publicly is a separate task.

### Verify and reproduce

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest
node --check stocksel/web/static/app.js
python -m stocksel.experiments.reentry  # one pre-registered synthetic comparison, not a search
```

The suite preserves the original leakage/sanity tests and adds feed, freshness,
API, valuation chronology, promotion and execution regressions. The experiment
logs both configurations (and failures) in an append-only registry under
`artifacts/gpt_review/`; a small evidence export is committed under
`docs/reentry_review/`. Existing research registries and used holdout locks must
be retained. Full research writes a safe `validation_summary.json` for the GUI;
quick runs cannot qualify for validation promotion. Missing metrics remain unavailable.
Future validation requires a versioned real point-in-time dataset, a newly
pre-registered untouched test window, and prospective paper trading. Synthetic
profit figures are software diagnostics, not an expectation of investment returns.

מערכת Python מלאה שעונה לכל מניה על ארבע שאלות: **האם לקנות, כמה, מתי להיכנס ומתי לצאת**. היא בנויה סביב שכבת ולידציה שמניחה כברירת מחדל ש**אין יתרון על פני קנייה והחזקה של המדד**, עד שהוכח אחרת: מחוץ למדגם, נטו מעלויות, ואחרי תיקון לריבוי ניסיונות.

> ⚠️ אין כאן הבטחת רווח. כל התוצאות בדוח הדוגמה מגיעות מ**שוק סינתטי** (הגישה לנתוני שוק חסומה בסביבת הפיתוח). הן מוכיחות שהצנרת עובדת, ולא שיש יתרון בשוק האמיתי. התוכנית, ההנחות, הסיכונים ומה שלא ניתן לאמת מתועדים ב-[`docs/PLAN.md`](docs/PLAN.md). תוצאות הרצת הדוגמה, כולל עולם ביקורת בלי אות וה-holdout, נמצאות ב-[`docs/RESULTS.md`](docs/RESULTS.md) ובדוחות ב-[`docs/sample_reports/`](docs/sample_reports/).

## התקנה

```bash
pip install -r requirements.txt          # אופציונלי: pip install anthropic yfinance xgboost
python -m pytest                          # 39 בדיקות, כולל דליפת מידע ובדיקה מקצה לקצה
```

## שימוש

```bash
python -m stocksel research               # walk-forward + כל הוולידציה + דוח HTML ב-reports_out/
python -m stocksel research --quick       # בלי CPCV, תוויות מעורבבות וחלק מבדיקות הרגישות
python -m stocksel holdout                # פותח את ה-holdout (24 חודשים) פעם אחת בלבד
python -m stocksel recommend [--asof DATE]   # המלצות מהקונפיגורציה הקפואה
python -m stocksel paper submit --file reports_out/recommendations_<date>.csv --asof <date>
python -m stocksel paper update           # מילוי פקודות בפתיחה הבאה, סטופים, עקומת הון
python -m stocksel monitor                # השוואה שבועית מול הבקטסט, drift בפיצ'רים, kill switch
python -m stocksel review                 # סקירה רבעונית של דעיכת אותות
```

כל הפרמטרים נמצאים בקובץ אחד: [`config.yaml`](config.yaml) (שוק, הון, drawdown נסבל, ספק נתונים, יקום, אותות, מודלים, ולידציה, עלויות, תיק, ביצוע, סיכון).
מפתחות API נקראים **רק** ממשתני סביבה (`NASDAQ_DATA_LINK_API_KEY`, `POLYGON_API_KEY`, `TIINGO_API_KEY`, `ANTHROPIC_API_KEY`).

## מבנה

| מודול | תפקיד |
|---|---|
| `stocksel/data/` | `DataBundle`, ספקים (Sharadar מומלץ, Polygon, Tiingo, yfinance עם דגל הטיה, CSV), שוק סינתטי עם אמת ידועה, יקום/נזילות, חיבור point-in-time |
| `stocksel/features/` | מומנטום 12-1/6-1, היפוך חודשי, Amihud, תנודתיות ו-idio vol, value, quality, SUE/PEAD, תחזיות, משטר שוק, סנטימנט חדשות (lexicon / Claude / קובץ), נרמול חתך-רוחבי |
| `stocksel/models/` | תוויות כיוון (תשואה עודפת), לוגיסטית / Ridge / ElasticNet, LightGBM, MLP, XGBoost אופציונלי, כיול Platt/isotonic, אנסמבל עם שער מקוון, SHAP |
| `stocksel/validation/` | walk-forward, Purged+Embargoed K-Fold, CPCV, רישום ניסיונות, PSR/DSR, PBO, holdout נעול, בדיקות שפיות ורגישות, דעיכת אותות, דגלים אדומים |
| `stocksel/backtest/` | מודל עלויות (עמלה + spread לפי נזילות + slippage + impact בשורש ריבועי), בקטסט חודשי מהיר, מנוע יומי (פתיחה הבאה, סטופים, delisting, kill switch), אסטרטגיית חדשות יומית |
| `stocksel/portfolio/` | Fractional Kelly ≤ ¼ אחרי haircut, vol targeting, תקרות פוזיציה/סקטור/חשיפה, kill switch |
| `stocksel/live/` | המלצות, paper trading, ניטור drift (PSI/KS), השוואה שבועית, kill switch |
| `stocksel/reports/` | דוח HTML עצמאי |

## מה המערכת אוכפת

- **אפס דליפה.** פיצ'ר בתאריך t משתמש רק במידע שהיה זמין בסגירה של t. דוחות כספיים נכנסים לפי `publish_date`, הכרזות וחדשות לפי חותמת הזמן. הביצוע בפתיחה הבאה. ה-holdout נחתך פיזית מהנתונים, ו-`tests/test_leakage.py` בודק כל כלל בנפרד.
- **בלי הטיית שורדים.** מניות שנמחקו נשארות בנתונים, כולל תשואת delisting. ספק שאינו survivorship-free מסומן כדגל אדום.
- **בלי מיקרו-קאפ.** סף שווי שוק (מינימום שאי אפשר להוריד מתחת ל-300M$), מחזור יומי, וחיתוך העשירון התחתון. בדיקת רגישות מריצה את האסטרטגיה רק על החצי הגדול של היקום.
- **רישום כל ניסיון,** כולל כישלונות. DSR ו-PBO מחושבים מעל הרישום המלא, והקונפיגורציה הסופית נקבעת מראש.
- **מודל מורכב נכנס לאנסמבל** רק אם הוא מכה את הבסיס הלוגיסטי מחוץ למדגם, נטו (בוטסטרפ), ורק על חודשים שכבר התממשו.
- **Haircut של 50%** לכל אות שפורסם, גם בתשואה הצפויה וגם בהסתברות שנכנסת ל-Kelly.
- **LLM:** אות החדשות קיים רק אחרי ה-`knowledge_cutoff` של המודל. אם ה-cutoff לא הוגדר, המערכת מניחה שהוא היום ולא מאפשרת בדיקה היסטורית.
- **דגלים אדומים אוטומטיים:** דיוק מעל 65%, Sharpe מעל 2, DSR מתחת ל-0.95, PBO מעל 0.5, אי-ניצחון על המדד נטו.

## פלט המלצה

`ticker | calibrated_prob | prob_after_haircut | expected_excess_after_haircut | position_pct | entry_date | planned_exit | stop | top3_signals_shap | confidence`

רמת הביטחון מוגבלת ל-Low כל עוד המערכת לא עברה DSR ≥ 0.95 ואת ה-holdout.

## שלב חי

paper trading של 3-6 חודשים (`live.paper_min_months`), השוואה שבועית מול התפלגות הבקטסט, kill switch על drawdown או על סטייה סטטיסטית (z < −2), וסקירה רבעונית של IC מתגלגל ל-36 חודשים כדי לזהות אותות מתים (`features.disabled`).
