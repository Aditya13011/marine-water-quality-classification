# Demonstration Guide
## AI-Based Marine Water Quality Classification System
### 5–7 Minute Evaluator Demo Script

---

## Pre-Demo Checklist (2 minutes before)

- [ ] Open: https://marine-water-quality-classification-br2p4oim3bzhnwxedwql5.streamlit.app
- [ ] Open: https://github.com/Aditya13011/marine-water-quality-classification (browser tab 2)
- [ ] Open: `D:\water-quality-prediction\` in File Explorer
- [ ] Have `python -m pytest tests/ -q` ready to run in terminal
- [ ] Ensure `.venv` is activated locally if running live tests

---

## Recommended Navigation Order

```
Home Page (30 s)
    → Prediction Page (90 s) ← most interactive
    → Data Analytics Page (60 s)
    → Model Performance Page (90 s)
    → About Project Page (30 s)
    → GitHub Repository (30 s)
    → (Optional) Terminal — pytest live (60 s)
```

---

## Page-by-Page Script

---

### 🏠 Page 1 — Home (30 seconds)

**What to show**: The five metric cards at the top.

**What to say**:
> "This is the home dashboard. The dataset contains **10,000 marine water samples** from a publicly archived Zenodo dataset — and I should point out upfront that it is **synthetic, mathematically generated data**, not real laboratory measurements. I've made this clear throughout the app.
>
> The selected model is a Random Forest with approximately **51.35% accuracy** — which is near random chance. That is honest and expected because the four input features I chose have negligible correlation with the target variable. I'll explain why shortly."

---

### 🔬 Page 2 — Prediction (90 seconds)

**What to show**: Enter two sample inputs and run predictions.

#### Sample Input 1 — Moderate Values

| Feature | Value | Range |
|---------|-------|-------|
| Temperature | 22.5 °C | 10–32 |
| pH | 7.8 | 6.5–8.8 |
| Salinity | 35.0 PSU | 30–40 |
| Turbidity | 15.0 NTU | 0.5–50 |

**Expected result**: Prediction is either "Safe" or "Unsafe" with probability ~50–55%. The result may vary — this illustrates the near-random nature of the model.

**What to say**:
> "On this page a user can adjust four sliders and get an instant prediction. I'll enter some mid-range values. Notice the probability bars — Safe and Unsafe are very close to 50/50. This is because the features I use — Temperature, pH, Salinity, Turbidity — genuinely have near-zero correlation with the target in this synthetic dataset."

#### Sample Input 2 — Boundary Test (show validation)

Set **Temperature = 9.0** (below minimum 10.0).

**Expected result**: Red error message: *"Temperature = 9.0 is outside the observed dataset range [10.0, 32.0] °C"*

**What to say**:
> "The app also performs input validation. If I enter a temperature of 9°C — below the dataset range — it shows an error instead of making an unreliable extrapolated prediction. This is a key safety feature."

---

### 📊 Page 3 — Data Analytics (60 seconds)

**Key charts to highlight**:

1. **Correlation Heatmap** (`03_correlation_heatmap.png`)

**What to say**:
> "This correlation heatmap is central to understanding the project. Look at the `is_safe` column. The threshold-defining columns — Dissolved Oxygen, Nitrate, Chlorophyll-a — show correlations of **0.7 to 0.9** with the target. That is target leakage. My four safe features — Temperature, pH, Salinity, Turbidity — show correlations below **0.01**. This is why Scenario A gives ~51% accuracy."

2. **Leakage Analysis** (`05_threshold_leakage_analysis.png`)

**What to say**:
> "This chart shows the distribution of the three threshold columns split by class. You can see a perfect bimodal separation — Safe samples have high dissolved oxygen, low nitrate, low chlorophyll-a. This is because the target is literally defined by hard thresholds on those three values."

---

### 📈 Page 4 — Model Performance (90 seconds)

**Key sections to highlight**:

1. **Scenario A Metrics Table** — show the ~51% row for Random Forest

**What to say**:
> "Here are the actual test-set evaluation metrics. All three models — Logistic Regression, Decision Tree, and Random Forest — score around 50–51%. Random Forest was selected with **51.35% accuracy** and **51.41% ROC-AUC**. These results are honest. I could have inflated them by using Scenario B, but that would be misleading."

2. **Scenario B Warning** — point to the red warning banner

**What to say**:
> "Scenario B uses all seven features including the threshold-defining ones. It gives ~97% accuracy — but I've clearly labelled it as a data leakage demonstration, not genuine performance. This is the key academic contribution: showing the difference between a model that learns from real signal versus one that memorises a rule."

3. **SHAP Summary Plot**

**What to say**:
> "SHAP — SHapley Additive exPlanations — explains which features the model relies on. Temperature has the highest mean SHAP value, likely because it is physically correlated with Dissolved Oxygen through gas solubility physics. All SHAP values are close to zero, which is consistent with the near-random accuracy."

---

### ℹ️ Page 5 — About Project (30 seconds)

**What to say**:
> "The About page documents the full methodology, explains the two-scenario design, and includes the responsible-use disclaimer. I want to point to the disclaimer explicitly: this system must not be used for real-world water safety decisions. The model was trained on synthetic data and achieves approximately 51% accuracy."

---

### 💻 GitHub Repository (30 seconds)

Open: https://github.com/Aditya13011/marine-water-quality-classification

**What to show**:
- The README renders with all sections including the performance table.
- Show `src/`, `tests/`, `pages/`, `reports/` folders.
- Show that no `.joblib` files are in the repository.

**What to say**:
> "The complete source code is on GitHub. Model files are excluded from the repository because they are large binaries — they are regenerated from the raw dataset automatically when the app first starts on Streamlit Cloud."

---

### 🧪 Optional — Live Test Run (60 seconds, if asked)

In the terminal:

```powershell
cd D:\water-quality-prediction
.\.venv\Scripts\Activate.ps1
python -m pytest tests/ -q
```

**Expected output**:
```
143 passed, 2 warnings in ~4s
```

**What to say**:
> "The project has 143 automated tests covering dataset integrity, prediction correctness, metric validation, and project structure. All tests pass with zero failures."

---

## Likely Evaluator Questions and Suggested Responses

| Question | Suggested Response |
|----------|-------------------|
| "Why is your accuracy only 51%?" | "Because I deliberately excluded the columns that define the target label, to avoid data leakage. The remaining features have near-zero correlation with the target in this synthetic dataset. 51% is the correct, honest result." |
| "What would happen if you used all features?" | "Accuracy would jump to ~97% — but that is Scenario B, which I show separately and clearly label as a data-leakage demonstration. It looks impressive but is not genuine learning." |
| "Is this model usable in real life?" | "No. It is trained on synthetic data and achieves ~51% accuracy. It cannot be used for real marine water safety assessments. This is stated clearly in the responsible-use disclaimer." |
| "How does it work on Streamlit Cloud without the model file?" | "The model `.joblib` file is excluded from git because it is 12 MB. At first startup, a script called `setup_model.py` automatically trains the model from the committed raw CSV. This takes ~60–120 seconds and only happens once." |
| "What is SHAP?" | "SHAP stands for SHapley Additive exPlanations. It uses game theory to calculate how much each feature contributed to a specific prediction. I used it to show that Temperature has the highest influence, likely because of its indirect physical correlation with dissolved oxygen." |
| "What does the confusion matrix show?" | "It shows predicted vs actual classes in a 2×2 table. My model's confusion matrix is nearly equal in all four cells — which means it is predicting essentially at random, consistent with the ~51% accuracy." |
| "How did you handle missing values?" | "I included a `SimpleImputer` in the pipeline with median strategy. The dataset has zero missing values, so it does not transform the data in practice, but it makes the pipeline production-safe against any future input with NaNs." |
| "Can you explain cross-validation?" | "I did not use cross-validation in this project — I used a single 80/20 stratified train/test split with seed 42. Cross-validation would give a more reliable estimate of performance but is more computationally expensive. Given the near-random performance, a single split is sufficient for this academic exercise." |

---

## Key Numbers to Remember

| Metric | Value |
|--------|-------|
| Dataset records | 10,000 |
| Class balance | 50% Safe / 50% Unsafe |
| Train size | 8,000 (80%) |
| Test size | 2,000 (20%) |
| Random seed | 42 |
| Selected model | Random Forest |
| Scenario A accuracy | **51.35%** |
| Scenario A F1-score | **50.83%** |
| Scenario A ROC-AUC | **51.41%** |
| Scenario B accuracy | **~97%** (leakage — NOT genuine) |
| Number of trees | 200 |
| Pytest tests | **143 passed, 0 failed** |
| Input features (Scenario A) | Temperature, pH, Salinity, Turbidity |
| Excluded features | Dissolved Oxygen, Nitrate, Chlorophyll-a |
