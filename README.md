# stocksel — מערכת מחקר ספקנית לבחירת מניות

מערכת Python מלאה שעונה לכל מניה על ארבע שאלות: **האם לקנות, כמה, מתי להיכנס ומתי לצאת**. היא בנויה סביב שכבת ולידציה שמניחה כברירת מחדל ש**אין יתרון על פני קנייה והחזקה של המדד**, עד שהוכח אחרת: מחוץ למדגם, נטו מעלויות, ואחרי תיקון לריבוי ניסיונות.

> ⚠️ אין כאן הבטחת רווח. כל התוצאות בדוח הדוגמה מגיעות מ**שוק סינתטי** (הגישה לנתוני שוק חסומה בסביבת הפיתוח). הן מוכיחות שהצנרת עובדת, ולא שיש יתרון בשוק האמיתי. התוכנית, ההנחות, הסיכונים ומה שלא ניתן לאמת מתועדים ב-[`docs/PLAN.md`](docs/PLAN.md).

## התקנה

```bash
pip install -r requirements.txt          # אופציונלי: pip install anthropic yfinance xgboost
python -m pytest                          # 36 בדיקות, כולל בדיקות דליפת מידע
```

## שימוש

```bash
python -m stocksel research               # walk-forward + כל הוולידציה + דוח HTML ב-reports_out/
python -m stocksel research --quick       # בלי CPCV, תוויות מעורבבות וחלק מבדיקות הרגישות
python -m stocksel holdout                # פותח את ה-holdout (24 חודשים) פעם אחת בלבד
python -m stocksel recommend [--asof DATE]   # המלצות מהקונפיגורציה הקפואה
python -m stocksel paper submit --file reports_out/recommendations_<date>.csv --asof <date>
python -m stocksel paper update           # מילוי פקודות בפתיחה הבאה, סטופים, עקומת הון
python -m stocksel monitor                # השוואה שבועית מול הבקטסט + kill switch
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
