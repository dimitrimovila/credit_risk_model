## Home Credit Default Risk 

### 01 — Exploratory Data Analysis (`01_eda_and_sql.ipynb`)

Exploratory analysis of all seven source tables using DuckDB SQL and Python
visualisations. Surfaces the patterns that drive the feature engineering phase.

**Coverage:**

| Block | Content |
|---|---|
| 1 | Data discovery — structure, column types, missing value profile |
| 2 | Target variable — 8.07% default rate, class imbalance implications |
| 3 | Numerical features — distributions, outliers, sentinel values |
| 4 | Categorical features — default rate per category, spread analysis |
| 5 | EXT_SOURCE scores — strongest individual predictors in the dataset |
| 6 | Bureau & bureau balance — DPD history, overdue amounts, prolongations |
| 7 | Previous applications — approval/refusal rates, credit amounts |
| 8 | Installment payments — payment delay, underpayment ratio |
| 9 | POS cash & credit card — DPD buckets, utilisation rate |

**Key findings:**
- `EXT_SOURCE_2` and `EXT_SOURCE_3` are the strongest univariate predictors
- `DAYS_EMPLOYED = 365243` is a sentinel for retirees/unemployed — requires cleaning
- Missing value patterns are structural, not random — missingness itself is predictive
- Clients with any DPD history in bureau show meaningfully higher default rates
- Credit card utilisation above 100% correlates with ~2× the baseline default rate 

---

### 02 — Feature Engineering (`02_feature_engineering.ipynb`)

End-to-end feature engineering pipeline across all 7 dataset tables,
implemented entirely in DuckDB SQL to handle the full data volume
(up to 27M rows) without memory issues.

**Pipeline:**

| Block | Description | Output |
|---|---|---|
| 1 | Cleaning & anomaly fixing | `app_train_clean` |
| 2 | Main table features (ratios, temporal, EXT_SOURCE combinations) | `app_train_features` |
| 3a-f | Aggregations from 6 secondary tables | `bureau_features`, `installments_features`, ... |
| 3g | Final join of all feature tables | `app_train_final` |
| 4 | Cross-table features (debt exposure, behavioural consistency) | `app_train_final_v2` |
| 5 | Categorical encoding (one-hot, ordinal, target encoding setup) | `app_train_final_v3` |
| Test | Same pipeline applied to `app_test` | `app_test_final` |

**Key engineering decisions:**
- `DAYS_EMPLOYED = 365243` (sentinel for "not employed") → `NULL` + `IS_NOT_EMPLOYED` flag
- All `AMT_*` columns capped at p99.9% computed on train only — no leakage to test
- Recency windows (6M, 12M) computed for all secondary tables — recent behaviour
  outweighs distant history
- Cross-table features combine signals across tables (e.g. `EXT_SCORE_ADJ_REFUSAL`
  adjusts external credit score based on HC's own refusal history)
- `OCCUPATION_TYPE` and `ORGANIZATION_TYPE` left as raw strings — target encoded
  inside the k-fold loop at modelling time to prevent leakage
- All materialised as DuckDB tables on disk — survives kernel restarts

**Result:** 247 features from 122 original columns across 7 tables.

---

### 03 — Modelling (`03_modelling.ipynb`)

Full modelling pipeline from feature-engineered tables to submission file.

**Pipeline:**

| Step | Description | OOF AUC |
|---|---|---|
| Baseline | LightGBM default params, 5-fold CV | 0.78680 |
| Feature selection | Drop 21 zero-importance features | 0.78692 |
| Hyperparameter tuning | Optuna Bayesian optimisation, 50 trials | 0.79061 |
| Stacking | LightGBM + XGBoost + CatBoost + LR meta-learner | **0.79165** |

**Key implementation details:**
- Stratified 5-fold cross-validation throughout — preserves 8.07% positive rate
- `scale_pos_weight = 11.4` handles class imbalance natively in all three base models
- Smoothed target encoding (`smooth=20`) fitted on train folds only — never on
  full dataset — prevents target leakage on `OCCUPATION_TYPE` and `ORGANIZATION_TYPE`
- Optuna search runs on a stratified 20% sample for speed, final evaluation on full data
- Meta-learner (Logistic Regression) trained on OOF predictions only — no leakage
  between stacking stages
- SHAP analysis provides both global interpretation (summary plot) and local
  explanation (waterfall plots for true positive and false negative cases)

**Meta-learner weights:**

| Model | Weight |
|---|---|
| LightGBM | 0.587 |
| CatBoost | 0.340 |
| XGBoost | 0.184 |

**SHAP findings:**
- Top 2 features are engineered cross-table features (`EXT_SCORE_VS_INTERNAL_DPD`,
  `EXT_SCORE_ADJ_REFUSAL`) — not raw data columns
- `CREDIT_TERM` shows a non-linear U-shape relationship with default risk —
  both very short and very long loan terms increase risk
- False negatives have a clear structure: clients who look reliable on paper
  (good external score, stable employment) but default due to unforeseen events —
  irreducible error that no historical feature can anticipate