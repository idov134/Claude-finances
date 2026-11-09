# מפת התיעוד

הפרויקט נכתב בשלושה סבבים: Claude (סבב 1), Codex/GPT (סבב 2) ו-Claude (סבב 3). כאן מרוכזים כל המסמכים, לפי השאלה שהם עונים עליה.

## להתחיל כאן
| שאלה | מסמך |
|---|---|
| מה המערכת, איך מריצים | [`../README.md`](../README.md) |
| מה התוכנית, ההנחות, הסיכונים ומה אי אפשר לאמת | [`PLAN.md`](PLAN.md) |
| מה התוצאות (סינתטי) | [`RESULTS.md`](RESULTS.md) |
| מי עשה מה, ומה כל אחד מצא אצל השני | [`CLAUDE_VS_CODEX.md`](CLAUDE_VS_CODEX.md) |

## מחקר וראיות
| נושא | מסמכים | מי |
|---|---|---|
| מחקר אלגוריתם: 6 שינויים, אישור על עולמות חדשים | [`algo_research_v2/`](algo_research_v2/) ([פרוטוקול](algo_research_v2/PROTOCOL.md), [תוצאות](algo_research_v2/RESULTS.md)) | Claude |
| חזית המנדטים ואבחון השער | [`algo_research_v3/`](algo_research_v3/) ([פרוטוקול](algo_research_v3/PROTOCOL.md), [תוצאות](algo_research_v3/RESULTS.md)) | Claude |
| **מבט ראשון על נתונים אמיתיים** (Yahoo, מוטה שורדים): אין כישרון, לא מכה את המדד | [`real_firstlook/`](real_firstlook/) ([פרוטוקול](real_firstlook/PROTOCOL.md), [תוצאות](real_firstlook/RESULTS.md)) | Claude |
| תיקון "בריחה למזומן" (מירכוז הסתברויות) | [`algo_research_v4/`](algo_research_v4/) ([פרוטוקול](algo_research_v4/PROTOCOL.md), [תוצאות](algo_research_v4/RESULTS.md)) | Claude |
| היסטוריית כל הניסיונות (בסיס ל-DSR/PBO) | [`research_history/`](research_history/) | Claude |
| ניסוי re-entry אחרי kill switch | [`ALGO_REVIEW.md`](ALGO_REVIEW.md), [`reentry_review/`](reentry_review/) | Codex |
| השוואה: logit מול פקטור שקוף מול משקל שווה מול מזומן | [`research_comparison/`](research_comparison/) | Codex |
| דוחות HTML לדוגמה | [`sample_reports/`](sample_reports/) | שניהם |

## הנדסה, מוצר ונכונות
| נושא | מסמך | מי |
|---|---|---|
| ביקורת, קריטריוני קבלה ומפת דרכים | [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md) | Codex |
| חוזי חשבונאות ונתונים (paper, עלויות, PIT) | [`FINANCIAL_CORRECTNESS.md`](FINANCIAL_CORRECTNESS.md) | Codex |
| אימות בפועל: בדיקות, דפדפן, ביצועים | [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md) | Codex |
| מקורות ומתחרים | [`COMPETITOR_RESEARCH.md`](COMPETITOR_RESEARCH.md) | Codex |
| צילומי מסך של ה-GUI | [`screenshots/`](screenshots/) | Codex |

## כללים שחלים על כל המחקר
1. **פרוטוקול לפני הרצה.** כל ניסוי נכתב ונדחף ל-git לפני שרץ, וכל ניסיון (כולל כישלונות) נרשם ב-`registry.jsonl`.
2. **DSR ו-PBO מחויבים על כל ההיסטוריה,** לא רק על הניסוי האחרון.
3. **holdout נפתח פעם אחת.** ה-holdout הסינתטי (2024–2025) נוצל. ה-holdout האמיתי (נוב' 2024–2026) נפתח פעמיים: בפעם השנייה אחרי תיקון באג נתונים, עם אסטרטגיה קפואה, ולכן הוא **מסומן מזוהם**. מחקר אמיתי עתידי צריך חלון חדש.
4. **שיפור נכנס לברירת המחדל** רק אם עבר גם בפיתוח וגם באישור על נתונים שלא נראו. עד כה אף שיפור בבחירת מניות לא עבר.
5. **כישרון נמדד ב-Alpha DSR,** לא רק ב-DSR. Sharpe של תיק long-only נובע ברובו מהחשיפה לשוק.
6. **שום תוצאה כאן אינה הבטחת רווח.**
