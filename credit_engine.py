import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split

def train_credit_model():
    """Generates synthetic credit data and trains an XGBoost Risk Model."""
    
    # 1. Generate Synthetic Financial Data
    np.random.seed(42)
    n_samples = 2000
    
    credit_score = np.random.normal(650, 80, n_samples).clip(300, 850)
    income = np.random.normal(65000, 25000, n_samples).clip(20000, 200000)
    utilization = np.random.normal(0.4, 0.2, n_samples).clip(0, 1)
    dti_ratio = np.random.normal(0.3, 0.15, n_samples).clip(0, 1)
    missed_payments = np.random.poisson(0.5, n_samples).clip(0, 5)
    
    # Define underlying logic for default risk (higher risk = higher chance of default)
    risk_score = (
        -0.005 * credit_score + 
        -0.00002 * income + 
        3.0 * utilization + 
        2.5 * dti_ratio + 
        1.2 * missed_payments
    )
    
    # Convert risk score to a binary default outcome (0 = Paid, 1 = Defaulted)
    probabilities = 1 / (1 + np.exp(-risk_score + 2)) # Sigmoid function
    default_status = (np.random.rand(n_samples) < probabilities).astype(int)
    
    df = pd.DataFrame({
        'Credit_Score': credit_score,
        'Annual_Income': income,
        'Credit_Utilization': utilization,
        'Debt_to_Income': dti_ratio,
        'Missed_Payments': missed_payments
    })
    
    # 2. Train the XGBoost Model
    X_train, X_test, y_train, y_test = train_test_split(df, default_status, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    
    # 3. Setup SHAP Explainer
    explainer = shap.TreeExplainer(model)
    
    return model, explainer, X_train.columns

def predict_and_explain(model, explainer, feature_names, customer_data):
    """Runs inference on a new customer and calculates feature importance."""
    df_customer = pd.DataFrame([customer_data], columns=feature_names)
    
    # Get Probability of Default
    probability = model.predict_proba(df_customer)[0][1]
    
    # Get SHAP values for explainability
    shap_values = explainer.shap_values(df_customer)
    
    return probability, shap_values[0], df_customer.iloc[0]