# 🧠 SentioMetrics: Teen Mental Health & Social Media Impact

**SentioMetrics** is an end-to-end Machine Learning project designed to analyze and predict the impact of digital behavior—specifically social media usage and sleep patterns—on adolescent mental health. 

The project transitions from deep data exploration and predictive modeling in a Jupyter environment to a real-time, interactive **Streamlit Dashboard** for clinical and personal risk assessment.

---

## 📊 Model Performance Metrics
The core predictive engine (a hyperparameter-tuned **Random Forest Classifier**) achieved "Gold Standard" performance during the validation phase:

| Metric | Score | Insight |
| :--- | :--- | :--- |
| **Accuracy** | **100% (1.00)** | Perfect classification across the stratified test set. |
| **ROC-AUC** | **1.00** | Perfect separation between "At Risk" and "Low Risk" classes. |
| **Recall (At Risk)** | **1.00** | Zero False Negatives; the model successfully identifies all at-risk teens. |
| **F1-Score** | **1.00** | Perfect balance between precision and recall. |

---

## 🚀 Key Features
- **Deep EDA:** Visual mapping of the "Tipping Point" where social media usage significantly degrades mental health.
- **Predictive Engine:** Advanced ensemble modeling with Random Forest and XGBoost.
- **Interpretability:** Integrated **SHAP** (SHapley Additive exPlanations) to explain individual risk factors.
- **SentioMetrics Dashboard:** A premium Streamlit web application for real-time risk calculation and lifestyle recommendations.
- **Persistence:** Serialized model logic using `joblib` for seamless deployment.

---

## 🛠️ Technology Stack
- **Languages:** Python 3.x
- **Data Science:** Pandas, NumPy, Scikit-Learn, XGBoost
- **Visualization:** Matplotlib, Seaborn
- **Interpretability:** SHAP
- **Web Framework:** Streamlit
- **Deployment/Persistence:** Joblib

---

## 📂 Project Structure
- `Mental_Health_Analysis.ipynb`: The complete data science pipeline (EDA -> Training -> Evaluation).
- `app.py`: The interactive Streamlit web application.
- `mental_health_model.joblib`: The serialized model and metadata (generated from the notebook).
- `Teen_Mental_Health_Dataset.csv`: The behavioral dataset.
- `plans.md`: Detailed project roadmap and motivation.
- `requirements.txt`: Environment dependencies.

---

## 🏃 How to Run

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Training & Persistence
Open `Mental_Health_Analysis.ipynb` and run all cells. This will perform the analysis and generate the `mental_health_model.joblib` file.

### 3. Launch the Dashboard
```bash
streamlit run app.py
```

---

## 💡 Motivation
The digital landscape has shifted adolescent behavior. **SentioMetrics** moves beyond anecdotes, using data to identify thresholds (like social media usage exceeding 4 hours/day) that correlate with increased anxiety and depression risk, providing actionable insights for parents, educators, and clinicians.
