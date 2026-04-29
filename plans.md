# Teen Mental Health & Social Media Impact: Predictive Modeling Pipeline (SentioMetrics)

## 1. Project Overview & Motivation

**Why are we doing this project?**
The modern digital landscape has fundamentally shifted adolescent behavior, with social media platforms like TikTok and Instagram becoming central to daily life. Concurrently, there has been a significant rise in reported teen mental health issues, including anxiety, stress, and depression. The motivation behind this project is to mathematically and empirically analyze this relationship. Instead of relying on anecdotes, we will use data science to uncover hidden patterns and identify the exact digital behaviors that correlate most strongly with mental health decline.

**What am I trying to achieve?**

1.  **Actionable Insights:** Generate a comprehensive Exploratory Data Analysis (EDA) report that visually maps how factors like `platform_usage`, `daily_social_media_hours`, and `sleep_hours` impact mental health metrics (`stress_level`, `anxiety_level`, `addiction_level`).
2.  **Predictive Capability:** Build a highly accurate, robust machine learning classification model (specifically a tree-based ensemble like XGBoost or Random Forest) capable of predicting a teenager's `depression_label` (Risk vs. No Risk) based on their demographic and behavioral data.
3.  **Interpretability:** Utilize advanced interpretability techniques (like Feature Importance and SHAP values) to explicitly rank the driving factors behind teen depression in this dataset, moving beyond a "black box" prediction to provide transparent, psychological insights.

---

## 2. Minute-Detail Execution Plan

### Phase 1: Environment Setup & Data Preprocessing

- **Objective:** Establish a clean, reproducible Python environment and format the raw dataset for machine learning.
- **Tasks:**
  1.  Initialize a virtual environment (`venv`) and install dependencies from `requirements.txt`.
  2.  Load `Teen_Mental_Health_Dataset.csv` using `pandas`.
  3.  **Data Auditing:** Check for missing values (NaNs), duplicates, and incorrect data types.
  4.  **Encoding Categorical Variables:**
      - Apply `LabelEncoder` for binary/ordinal categorical data (e.g., `gender`).
      - Apply `OneHotEncoder` (or `pd.get_dummies()`) for multi-class nominal data (e.g., `platform_usage` -> `Instagram`, `TikTok`, `Both`).
  5.  **Scaling (Optional for Trees, good for EDA):** Standardize continuous variables like `daily_social_media_hours` if distance-based calculations are needed later.

### Phase 2: Comprehensive Exploratory Data Analysis (EDA)

- **Objective:** Uncover the statistical story within the dataset through visualization.
- **Tasks:**
  1.  **Univariate Analysis:** Plot the distributions of all numerical variables (e.g., histograms for `sleep_hours`, `stress_level`) and count plots for categorical variables.
  2.  **Bivariate Analysis:**
      - Boxplots mapping `platform_usage` against `addiction_level`.
      - Scatter plots mapping `daily_social_media_hours` against `academic_performance`.
  3.  **Multivariate Analysis & Correlation:** Generate a Spearman/Pearson correlation heatmap using `seaborn` to identify highly correlated features (e.g., does high `screen_time_before_sleep` heavily correlate with low `sleep_hours` and high `depression_label`?).

### Phase 3: Feature Engineering & Selection

- **Objective:** Enhance the dataset to improve the model's predictive power.
- **Tasks:**
  1.  **Derived Metrics:** Create new features if logical (e.g., `total_daily_screen_time` = `daily_social_media_hours` + `screen_time_before_sleep`).
  2.  **Collinearity Check:** Remove redundant features to reduce multi-collinearity noise.
  3.  **Class Imbalance Check:** Evaluate the distribution of the target variable (`depression_label`). If heavily skewed, we will implement **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure the model doesn't become biased towards the majority class.

### Phase 4: Model Selection, Training & Hyperparameter Tuning

- **Objective:** Train state-of-the-art machine learning algorithms and rigorously optimize them.
- **Tasks:**
  1.  **Data Splitting:** Split data into 80% Training and 20% Testing sets using a stratified split (to maintain class ratios).
  2.  **Baseline Modeling:** Train baseline models (Logistic Regression, Random Forest, XGBoost) with default parameters to establish a performance floor.
  3.  **Advanced Tuning:** Utilize **Optuna** (Bayesian Optimization) or `GridSearchCV` on the best performing model (likely XGBoost or Random Forest).
      - _Tuning Parameters:_ `max_depth`, `learning_rate`, `n_estimators`, `subsample`.
  4.  **Cross-Validation:** Implement 5-fold Stratified Cross-Validation to ensure the model generalizes well to unseen data and isn't overfitting.

### Phase 5: Evaluation & Interpretability

- **Objective:** Grade the model's real-world viability and explain _how_ it makes decisions.
- **Tasks:**
  1.  **Classification Metrics:** Generate a Classification Report containing Precision, Recall, F1-Score, and overall Accuracy. Focus heavily on **Recall** (we don't want to miss a depressed teen/false negative).
  2.  **Confusion Matrix:** Visualize True Positives, True Negatives, False Positives, and False Negatives.
  3.  **ROC-AUC Curve:** Plot the Receiver Operating Characteristic curve to measure the model's ability to distinguish between classes.
  4.  **SHAP Analysis:** Generate SHAP (SHapley Additive exPlanations) summary plots.
  5.  **Tipping Point Analysis:** Use SHAP dependence plots to identify the "threshold" where social media hours significantly increase mental health risks.

### Phase 6: Practical Risk Assessment Tool
- **Objective:** Bridge the gap between data science and real-world application.
- **Tasks:**
  1.  Develop a Python-based **Risk Calculator** function that takes real-time user inputs.
  2.  Implement probability-based recommendations (e.g., Clinical Referral vs. Lifestyle Adjustment).
  3.  Validate the tool against simulated cases to ensure reliability.

### Phase 7: Web Application Deployment
- **Objective:** Create a user-facing dashboard for real-time risk assessment.
- **Tasks:**
  1.  Develop `app.py` using Streamlit.
  2.  Integrated custom CSS for a premium UI/UX.
  3.  Implement model loading and feature mapping for seamless inference.

### Phase 8: Final Reporting & Consolidation

- **Objective:** Package the findings into a professional format.
- **Tasks:**
  1.  Consolidate the optimized, final pipeline into a clean Jupyter Notebook (`mental_health_analysis.ipynb`).
  2.  Export key visual findings (SHAP charts, correlation matrices) into an `assets` folder.
  3.  Write a final executive summary detailing the psychological insights uncovered by the algorithm.

---

## 3. Technology Stack & Rationale

These libraries will be strictly version-controlled in our `requirements.txt`:

- **Data Manipulation:**
  - `pandas`: For robust DataFrame manipulation, handling CSVs, and data cleaning.
  - `numpy`: For high-performance vectorized mathematical operations.
- **Visualization:**
  - `matplotlib`: The foundational plotting library.
  - `seaborn`: Built on matplotlib, allows for beautiful, statistically-informed visualizations (heatmaps, violin plots) with minimal code.
- **Machine Learning Pipeline:**
  - `scikit-learn`: For data splitting, scaling, encoding, baseline modeling (Random Forest), and evaluation metrics (confusion matrix, ROC).
  - `xgboost`: An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. It will likely be our champion model.
  - `imbalanced-learn` (`imblearn`): Specifically to utilize SMOTE, ensuring our model handles any imbalance in the `depression_label` fairly.
- **Hyperparameter Optimization:**
  - `optuna`: A next-generation hyperparameter optimization framework that uses Bayesian methods to find the optimal model configuration much faster than traditional Grid Search.
- **Model Interpretability:**
  - `shap`: The gold standard for interpreting machine learning models. It will allow us to explain the exact impact of every single feature on the final depression prediction.
- **Environment:**
  - `jupyterlab` / `notebook`: For interactive coding, allowing us to see charts and data frames immediately within the workflow.
