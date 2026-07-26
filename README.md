# Credit Default Prediction & Risk Explainability Engine

An institutional-grade credit underwriting and risk classification engine powered by XGBoost. The system predicts customer Probability of Default (PoD) and integrates SHAP (SHapley Additive exPlanations) to provide local feature attribution and regulatory decision explainability.

##  Core Features

* **Gradient Boosted Decision Trees:** Implements XGBoost classification with custom scale weighting (`scale_pos_weight`) to handle financial class imbalance.
* **Explainable AI (XAI) Integration:** Uses SHAP TreeExplainer to decompose complex model outputs into interpretable feature-level reason codes.
* **Interactive Risk Dashboard:** A Streamlit interface allowing risk teams to input customer profile metrics and receive real-time approval decisions alongside SHAP feature attributions.

##  Local Execution

1. Clone the repository:
   ```bash
   git clone [https://github.com/satyaCO7/credit-default-risk-explainability-engine.git](https://github.com/satyaCO7/credit-default-risk-explainability-engine.git)
   cd credit-default-risk-explainability-engine