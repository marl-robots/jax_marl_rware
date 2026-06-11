### מדדים חשובים (Must‑have)
- **Episode return (mean ± std)** — סכום הפרסים לכל אפיזודה; צורה: scalar.  
  ```python
  ep_returns = rraw_t.sum(axis=0).sum(axis=-1)  # [E]
  ep_return = ep_returns.mean()
  ep_return_std = ep_returns.std()
  ```
- **Per‑agent total return (mean ± std)** — מגלה חוסר איזון בין סוכנים; צורה: [N] או [E,N] לאגירה.  
  ```python
  per_agent_return = rraw_t.sum(axis=0)  # [E,N]
  per_agent_mean = per_agent_return.mean(axis=0)  # [N]
  per_agent_std  = per_agent_return.std(axis=0)   # [N]
  ```
- **Throughput / total deliveries per episode (mean ± std)** — סך מסירות לכל env; צורה: scalar.  
  ```python
  deliveries = deliv_t.sum(axis=0).sum(axis=-1)  # [E]
  deliveries_mean = deliveries.mean()
  deliveries_std = deliveries.std()
  ```
- **Per‑agent success / mean_num_successful** — ממוצע מספר סוכנים שהצליחו בכל env; צורה: scalar.  
  ```python
  success_per_agent = (deliv_t.sum(axis=0) > 0)  # [E,N]
  mean_num_successful = success_per_agent.sum(axis=-1).mean()
  ```
- **Team success rate** — אחוז envs שבהם כל הסוכנים הצליחו לפחות פעם אחת; צורה: scalar.  
  ```python
  team_success_rate = jnp.all(success_per_agent, axis=-1).mean()
  ```
- **Policy entropy (mean ± std)** — per‑agent entropy over time; צורה: scalar או [N].  
- **Actor / Critic losses (full vectors + mean/std)** — שמור וקטור אפוקים; אל תשתמש רק ב‑[-1].

---

### מדדים רצויים (Recommended)
- **Per‑step event rates (pickup/block/idle) mean ± std** — frac ו‑frac_std על [T,E,N].  
  ```python
  frac = lambda x: x.mean()
  frac_std = lambda x: x.std()
  ```
- **Time to first delivery / time to completion** — ממוצע ופרסנטילים; מזהה בעיות latency.  
- **Per‑env temporal statistics** — `E_per_T_mean` ו‑`E_per_T_std` כדי לזהות envs לא יציבים.  
  ```python
  E_per_T_mean = lambda x: x.mean(axis=0)  # [E,...] -> mean over time
  E_per_T_std  = lambda x: x.std(axis=0)
  ```
- **Percentiles (10/50/90) עבור return, deliveries, step_time** — נותן תמונה מעבר ל‑mean/std.  
  ```python
  p10, p50, p90 = jnp.percentile(ep_returns, jnp.array([10,50,90]))
  ```
- **Fairness / imbalance** — std או Gini על per‑agent deliveries/returns.  
- **KL divergence per update או gradient norms** — לניטור יציבות האימון.

---

### מדדים מתקדמים (Nice‑to‑have / Research)
- **RWARE / reward‑weighted metrics** — ממוצע תגמול משוקלל לפי הצלחה או לפי מספר סוכנים מצליחים.  
  ```python
  weights = team_success_per_env.astype(jnp.float32)
  rware = (ep_returns * weights).sum() / (weights.sum() + 1e-12)
  ```
- **Credit‑assignment proxies** — correlation בין פעולות לתוצאה, attribution metrics, Shapley‑style proxies.  
- **Robustness tests** — performance תחת רעש בתצפיות, השבתת סוכן, שינוי דינמיקה; מדד recovery time.  
- **Per‑agent action distribution / diversity index** — האם סוכנים מתואמים או מתחרים יתר על המידה.  
- **Confidence intervals / bootstrap** על מדדים מרכזיים לפני השוואות.

---

### מדדים ספציפיים ל‑multi‑agent algorithms (IPPO, IA2C, MA‑A2C, MAPPO)
- **Per‑agent advantage / TD error stats** — mean/std/skew; חשוב ל‑A2C‑type.  
- **KL between old/new policy (PPO)** — למדידת עדכון מדיניות; עקוב אחרי ערכים גדולים.  
- **Number of critic updates vs actor updates** — השפעה על value_loss; דווח ביחס ללמידה.  
- **Entropy per agent over training** — האם MAPPO/IPPO שומרים על חקירה מספקת.  
- **Per‑agent gradient norms** — לזהות סוכנים שמקבלים עדכונים גדולים מדי.

---

### איך לארגן ולדווח (Practical rules)
- **דווח תמיד mean ± std ופרסנטילים**; אל תסתפק ב‑mean בלבד.  
- **שמור וקטורי אפוקים** (`diag["epoch_*"]`) לניתוח מאוחר; אל תשתמש רק ב‑`[-1]`.  
- **אגרגציה על seeds**: הרץ מספר זרעים ודווח mean ± std across seeds לפני מסקנות.  
- **שמור time series לדגימות env/agent** (לפחות כמה envs) כדי לאתר outliers.  
- **השתמש ב‑eps ו‑ddof מודעים**: JAX `std()` משתמש ב‑ddof=0; אם רוצים sample std השתמשו ב‑(n-1).  
- **תגיות מטא‑מידע**: seed, env config, checkpoint, hyperparams — לכל מדד.

---

### בדיקת רשימה מהירה (Checklist)
**הוספת מיידית מומלצת:**  
- per‑agent return mean/std; percentiles for returns; save full epoch loss vectors; variance across seeds; KL or grad norm.

**אם יש זמן/משאבים:**  
- robustness experiments; credit assignment metrics; recovery time; Gini fairness; bootstrap CIs.

---
| **שם המדד** | **נוסחה קצרה** | **צורה** | **למה חשוב** |
| --- | --- | --- | --- |
| **episode_return (mean)** | ``ep_returns ``= ``rraw_t.sum(0).sum(-1); ``mean(ep_returns)`` | scalar | מדד ביצוע כולל של המדיניות; הבסיס להשוואות. |
| **episode_return (std)** | ``ep_returns.std()`` | scalar | מראה יציבות/שונות בביצועים בין אפיזודות/סביבות. |
| **per‑agent return (mean)** | ``per_agent ``= ``rraw_t.sum(0); ``per_agent.mean(axis=0)`` | ``[N]`` | חושף חוסר איזון בין סוכנים; מי מקבל את רוב הפרס. |
| **per‑agent return (std)** | ``per_agent.std(axis=0)`` | ``[N]`` | מזהה סוכנים לא יציבים או בעיות קרדיט. |
| **total deliveries (mean)** | ``deliv_t.sum(0).sum(-1).mean()`` | scalar | throughput של המערכת; מדד ביצוע משימה ישיר. |
| **total deliveries (std)** | ``deliv_t.sum(0).sum(-1).std()`` | scalar | מראה חוסר עקביות בין envs ביכולת למסור. |
| **mean_num_successful** | ``(deliv_t.sum(0)>0).sum(-1).mean()`` | scalar | ממוצע מספר סוכנים שהשיגו לפחות מסירה; מדד חלקי לקבוצה. |
| **team_success_rate** | ``all(deliv_t.sum(0)>0, ``axis=-1).mean()`` | scalar | אחוז הסביבות שבהן כל הסוכנים הצליחו; מדד הצלחה קבוצתי חזק. |
| **delivery_rate (per step)** | ``frac(deliv_indicator) ``= ``mean(indicator)`` | scalar | קצב אירועים בזמן; שימושי לניתוח דינמיקה בזמן אמת. |
| **time_to_first_delivery (mean)** | ``mean(first_delivery_time_per_env)`` | scalar | מדד latency; חשוב למשימות רגישות לזמן. |
| **policy entropy (mean/std)** | ``entropy_t.mean(), ``entropy_t.std()`` | scalar / scalar | מודד חקירה מול ניצול; איתות לבעיות קונברגנציה. |
| **actor / value loss (mean/std + vectors)** | ``diag["epoch_actor_loss"].mean(), ``.std()`` | scalar / vector | ניטור יציבות האימון; שמור וקטורים לאבחון. |
| **KL or grad norm** | ``mean(KL_per_update) ``or ``mean(grad_norms)`` | scalar | מזהה עדכונים גדולים או בעיות יציבות ב‑PPO/אופטימיזציה. |
| **percentiles (10/50/90) returns** | ``percentile(ep_returns,[10,50,90])`` | 3 scalars | מציג התפלגות מעבר ל‑mean/std; חושף זנבות/outliers. |
| **fairness / imbalance (Gini or std)** | ``std(per_agent_deliveries) ``or ``Gini(per_agent)`` | scalar | בודק חלוקת עבודה בין סוכנים; חשוב לפריסה הוגנת של משימות. |