# Viva Questions and Answers
## AI-Based Marine Water Quality Classification System

> All answers are specific to this project. Accuracy figures, dataset properties, and design choices reflect the actual implementation.

---

## Section 1 — Python and Streamlit

**Q1. What is Streamlit and why did you use it?**

Streamlit is an open-source Python framework for building interactive web applications purely in Python — no HTML, CSS, or JavaScript knowledge is required. I chose it because it lets data scientists build dashboards quickly. In this project, Streamlit powers four pages: prediction, data analytics, model performance, and about. It is deployed for free on Streamlit Community Cloud.

---

**Q2. What does `@st.cache_resource` do in your app?**

`@st.cache_resource` is a Streamlit decorator that tells Streamlit to run a function's body **only once per server lifecycle** and then cache the result in memory. I used it in `app.py` to wrap the model-loading/training bootstrap function. That way, the 12 MB Random Forest pipeline is loaded from disk once when the server starts, and reused on every subsequent page navigation or button click. Without it, the model would be reloaded from disk on every user interaction, making the app slow.

---

**Q3. What is a virtual environment and why did you use one?**

A virtual environment (`.venv` in this project) is an isolated Python installation that contains only the packages needed for this project. Without it, all Python projects on a machine share the same package versions, which often causes version conflicts. Using `.venv` ensures that `scikit-learn 1.3+`, `shap 0.44+`, and `streamlit 1.30+` are installed specifically for this project without affecting other programs.

---

**Q4. What is `sys.path.insert()` and why is it in your page files?**

`sys.path` is Python's list of directories where it searches for importable modules. On Streamlit Cloud, page files in the `pages/` folder are loaded with a different working directory than `app.py`. By inserting `src/` at the start of `sys.path`, I ensure that `import prediction` (from `src/prediction.py`) works correctly regardless of how or from where the page is run.

---

## Section 2 — Dataset and Preprocessing

**Q5. What is your dataset and where does it come from?**

The dataset is a publicly archived synthetic marine water quality dataset from **Zenodo (record 19506986)**. It was originally published on Kaggle by the author `robust_mariner`. It contains **10,000 records and 9 columns** representing physicochemical water quality parameters. The dataset is **mathematically derived** — not from real laboratory or field measurements.

---

**Q6. What does "synthetic dataset" mean? Why does it matter?**

Synthetic means the data was generated using mathematical equations (biogeochemical models) rather than measured from real ocean water samples. This matters critically for my project:
1. The target label (`is_safe`) is a **deterministic function** — there is no noise or ambiguity.
2. My ML model cannot be generalised to predict actual marine safety.
3. The model's accuracy (~51%) reflects the feature design, not a fault in the algorithm.

---

**Q7. What is a train/test split and why is it important?**

A train/test split divides the dataset into two non-overlapping parts: training data (used to fit the model) and test data (used to evaluate it on unseen examples). I used an **80/20 split** — 8,000 training samples, 2,000 test samples — with stratification (preserving the 50/50 class balance in both splits) and a random seed of 42 for reproducibility. Without a test split, we cannot know how the model performs on new, unseen data.

---

**Q8. What is StandardScaler and why did you use it?**

`StandardScaler` transforms each feature so it has **mean = 0** and **standard deviation = 1**. This is important because ML models like Logistic Regression are sensitive to feature scale — a feature in the range 30–40 (Salinity) would dominate a feature in the range 0.5–50 (Turbidity) without scaling. I fit the scaler on **training data only** and applied it to the test set, preventing data snooping.

---

**Q9. What is `SimpleImputer` and why is it in the pipeline?**

`SimpleImputer` fills in missing values. I used the `median` strategy — missing values in a column are replaced with that column's median value. My dataset has **no missing values** (verified in testing), so it does not change the data in practice. However, including it in the pipeline protects the deployed app against future input data that might have NaN values and ensures the pipeline is production-safe.

---

**Q10. What is a `sklearn Pipeline` and why did you use one?**

A `sklearn Pipeline` chains multiple processing steps into a single object. My pipeline has three steps: `SimpleImputer → StandardScaler → RandomForestClassifier`. The key benefit is that when `pipeline.predict(X)` is called with raw (unscaled) input, it automatically applies imputation and scaling before passing the data to the model. This prevents the common deployment mistake of forgetting to scale input at inference time.

---

## Section 3 — Classification

**Q11. What is a classification problem?**

Classification is a supervised machine learning task where the model learns to assign input samples to one of a fixed set of categories (classes). In my project, the model classifies marine water samples as either **Safe (class 1)** or **Unsafe (class 0)** — a binary classification problem.

---

**Q12. What is class balance and why does it matter?**

Class balance refers to how evenly the target classes are distributed. My dataset is **perfectly balanced — exactly 50% Safe and 50% Unsafe**. This matters because imbalanced datasets (e.g., 95% Safe, 5% Unsafe) can cause a model to simply predict the majority class and still appear accurate. With 50/50 balance, a model that always predicts "Safe" would get exactly 50% accuracy — so even my ~51.35% result represents a tiny improvement over the trivial baseline.

---

## Section 4 — Machine Learning Algorithms

**Q13. What is Logistic Regression and how does it work?**

Logistic Regression is a linear classification algorithm. It learns a weighted combination of input features and applies a **sigmoid function** to map the result to a probability between 0 and 1. Despite the word "regression," it is used for classification. It performs best when the decision boundary between classes is approximately linear. In my Scenario A, its accuracy was **50.40%** — slightly below Random Forest.

---

**Q14. What is a Decision Tree?**

A Decision Tree is a model that learns a series of if-else rules from training data, forming a tree structure. Each internal node tests a feature value, each branch represents an outcome, and each leaf makes a prediction. I used `max_depth=6` and `min_samples_leaf=20` to prevent the tree from memorising noise (overfitting). Its Scenario A accuracy was **50.10%**.

---

**Q15. What is a Random Forest and why was it selected?**

A Random Forest is an **ensemble** of many decision trees (200 in my project). Each tree is trained on a random bootstrap sample of the data, and each node split considers a random subset of features. The final prediction is a **majority vote** of all 200 trees. This approach reduces the variance and overfitting of individual trees. I selected it because it achieved the highest **F1-score (50.83%)** on the Scenario A test set and is the most compatible algorithm with SHAP's `TreeExplainer`.

---

**Q16. Why is Random Forest more robust than a single Decision Tree?**

A single decision tree can overfit — it memorises the training data, including noise, leading to poor performance on new data. Random Forest reduces this by:
1. **Bagging**: Training each tree on a different random sample.
2. **Feature randomness**: Each split considers only a random subset of features.
Averaging many imperfect trees produces a more stable, generalised prediction than any single tree.

---

## Section 5 — Evaluation Metrics

**Q17. What is accuracy and what is its limitation?**

Accuracy = (Correct predictions) / (Total predictions). My Random Forest (Scenario A) achieves **51.35%**. Its main limitation: on imbalanced datasets, a model that always predicts the majority class appears accurate (e.g., 95% accurate on a dataset that is 95% one class). In my balanced dataset, accuracy and F1-score are equivalent, so this limitation does not apply — but I still report all five metrics for completeness.

---

**Q18. What is the difference between Precision and Recall?**

- **Precision**: Of all samples predicted as Safe, what fraction were actually Safe? Measures false-positive rate. Formula: TP / (TP + FP).
- **Recall**: Of all actually Safe samples, what fraction did the model correctly identify? Measures false-negative rate. Formula: TP / (TP + FN).

In my project: Precision = **51.38%**, Recall = **50.30%** for Random Forest Scenario A — both very close to 50%, confirming near-random performance.

---

**Q19. What is F1-Score and why did you use it to select the best model?**

F1-Score is the **harmonic mean of Precision and Recall**: F1 = 2 × (P × R) / (P + R). It penalises models that have a large gap between precision and recall. I used F1 as the selection criterion because my dataset is perfectly balanced (so F1 ≈ accuracy) but F1 is more meaningful than accuracy when class balance is at risk of shifting in production.

---

**Q20. What is ROC-AUC?**

ROC-AUC (Receiver Operating Characteristic — Area Under Curve) measures a model's ability to **discriminate between classes across all probability thresholds**. A value of:
- **1.0** = perfect classifier
- **0.5** = random classifier (equivalent to coin flip)
- **< 0.5** = worse than random

My Random Forest Scenario A ROC-AUC = **51.41%** — barely above 0.5, confirming the near-random nature of the model on leakage-free features.

---

**Q21. What is a Confusion Matrix?**

A confusion matrix is a table showing **actual vs predicted classes**:

|  | Predicted Unsafe | Predicted Safe |
|--|---|---|
| **Actual Unsafe** | True Negative (TN) | False Positive (FP) |
| **Actual Safe** | False Negative (FN) | True Positive (TP) |

For my model: TN ≈ 495, FP ≈ 505, FN ≈ 494, TP ≈ 506 (approximately, from a ~51% accuracy on 2,000 balanced samples). The near-equal distribution in all four cells confirms random-chance prediction.

---

## Section 6 — Data Leakage

**Q22. What is data leakage and how does it affect your project?**

Data leakage occurs when information that would not be available at prediction time is accidentally included in the training features. In my dataset, `Dissolved_Oxygen`, `Nitrate`, and `Chlorophyll_a` **directly determine the target** via the safety rule. Including them gives the model a "cheat sheet" — it learns the rule instead of a pattern.

**Effect on Scenario B**: ~97–100% accuracy — not genuine.
**My solution**: Scenario A excludes these three columns entirely.

---

**Q23. How did you prevent data leakage in Scenario A?**

Three layers of protection:
1. **Feature definition**: `FEATURES_A = ["Temperature", "pH", "Salinity", "Turbidity"]` in `data_preprocessing.py` — the three threshold columns are simply not included.
2. **Training guard**: `train_model.py` asserts `assert not found_leakage_cols` before training Scenario A.
3. **Automated test**: `tests/test_dataset.py::TestTargetLeakageProperties` verifies Pearson |r| < 0.05 between each Scenario A feature and the target.

---

**Q24. Why is Scenario B's ~97% accuracy described as "not genuine"?**

Because the model is not learning any real pattern about water safety — it is learning to replicate the mathematical threshold rule embedded in the data. If you deploy this model on real marine water samples, it would fail completely because the relationship between Dissolved Oxygen/Nitrate/Chlorophyll-a and the safety label is specific to the synthetic generation process, not to actual marine chemistry.

---

## Section 7 — SHAP and Feature Importance

**Q25. What is SHAP and why did you use it?**

SHAP (**SHapley Additive exPlanations**) is a game-theory-based method for explaining individual model predictions. For each prediction, SHAP assigns each feature a value representing its **positive or negative contribution** to the prediction. I used `shap.TreeExplainer` (optimised for tree-based models) on the Scenario A Random Forest. Even though the model has low accuracy, SHAP still shows which features the model is relying on most — **Temperature** has the highest mean |SHAP| value, likely due to its indirect physical correlation with Dissolved Oxygen.

---

**Q26. What does the SHAP summary plot show?**

The SHAP beeswarm plot (`reports/shap_summary.png`) shows:
- **Y-axis**: Features ranked by mean absolute SHAP value (most important at top).
- **X-axis**: SHAP value — positive means "pushes prediction toward Safe," negative means "pushes toward Unsafe."
- **Colour**: Feature value (red = high, blue = low).

In my project, all SHAP values are very close to zero, which is consistent with the near-random model performance.

---

## Section 8 — Testing and Deployment

**Q27. What testing did you perform and what were the results?**

I used **pytest** for automated testing. The full test suite contains **143 tests across 4 test modules**:
- `test_dataset.py` — 18 tests: dataset integrity, class balance, leakage detection
- `test_prediction_pipeline.py` — 27 tests: pipeline loading, prediction output contracts, boundary validation
- `test_model_metrics.py` — 32 tests: Scenario A metric assertions, UI disclaimer checks
- `test_project_integrity.py` — 28 tests: file structure, requirements, `.gitignore`, syntax

**Result**: 143 passed, 0 failed — run with `python -m pytest tests/ -v`.

---

**Q28. How is the model deployed on Streamlit Community Cloud without uploading the `.joblib` file?**

The `.joblib` model file (12 MB) is excluded from git because it is a large binary. Instead, I created `src/setup_model.py` which:
1. Checks if the model file exists at startup.
2. If not, runs `data_preprocessing.py` and `train_model.py` as subprocesses.
3. These scripts regenerate the model from the committed raw CSV.

This runs **once** at first cold-start (~60–120 seconds), then the result is cached in memory by `@st.cache_resource`. Subsequent visits are instant.

---

## Section 9 — Limitations and Future Improvements

**Q29. What are the main limitations of your project?**

1. **Synthetic dataset**: The model cannot be used for real marine water safety decisions.
2. **~51% accuracy**: Near-random — the four Scenario A features have negligible correlation with the synthetic target.
3. **Deterministic target**: Any rule-based system would outperform ML here.
4. **Ephemeral cloud storage**: Model is regenerated on cloud restart.
5. **No real-world validation**: The pipeline has not been tested on real field data.

---

**Q30. What improvements would you make with more time?**

1. **Real dataset**: Apply the pipeline to a real marine monitoring dataset (e.g., from ICES or US EPA).
2. **Advanced models**: Explore XGBoost, LightGBM, or neural networks.
3. **Temporal analysis**: Add LSTM for time-series water quality prediction.
4. **DVC integration**: Version control for large model artifacts to avoid re-training on cloud restarts.
5. **CI/CD pipeline**: GitHub Actions to run pytest on every pull request.
6. **Model monitoring**: Track prediction distributions over time to detect drift.
