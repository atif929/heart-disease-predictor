"""
Utility Functions for Heart Disease Prediction App
"""

import pandas as pd
import numpy as np
import joblib
import os

def load_models():
    """Load all trained models and scaler"""
    models = {}
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Random Forest': 'random_forest.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'SVM': 'svm.pkl'
    }
    
    for name, filename in model_files.items():
        path = os.path.join('models', filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    
    # Load scaler
    scaler_path = os.path.join('models', 'scaler.pkl')
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    return models, scaler

def load_model_comparison():
    """Load model comparison results"""
    comparison_path = 'models/model_comparison.csv'
    if os.path.exists(comparison_path):
        return pd.read_csv(comparison_path)
    return None

def predict_disease(model, scaler, input_data):
    """
    Make prediction using the selected model
    
    Parameters:
    - model: trained model
    - scaler: fitted StandardScaler
    - input_data: dict with feature values
    
    Returns:
    - prediction: 0 or 1
    - probability: probability of disease
    """
    # Create DataFrame from input
    df = pd.DataFrame([input_data])
    
    # Scale features
    df_scaled = scaler.transform(df)
    
    # Make prediction
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0]
    
    return prediction, probability

def get_feature_descriptions():
    """Return descriptions for all features"""
    descriptions = {
        'age': 'Age of the patient in years',
        'sex': 'Gender (1 = Male, 0 = Female)',
        'cp': 'Chest Pain Type (0-3)',
        'trestbps': 'Resting Blood Pressure (mm Hg)',
        'chol': 'Serum Cholesterol (mg/dl)',
        'fbs': 'Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)',
        'restecg': 'Resting ECG Results (0-2)',
        'thalach': 'Maximum Heart Rate Achieved',
        'exang': 'Exercise Induced Angina (1 = Yes, 0 = No)',
        'oldpeak': 'ST Depression Induced by Exercise',
        'slope': 'Slope of Peak Exercise ST Segment (0-2)',
        'ca': 'Number of Major Vessels (0-3)',
        'thal': 'Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)'
    }
    return descriptions

def get_feature_ranges():
    """Return valid ranges for all features"""
    ranges = {
        'age': (29, 77),
        'sex': (0, 1),
        'cp': (0, 3),
        'trestbps': (94, 200),
        'chol': (126, 564),
        'fbs': (0, 1),
        'restecg': (0, 2),
        'thalach': (71, 202),
        'exang': (0, 1),
        'oldpeak': (0.0, 6.2),
        'slope': (0, 2),
        'ca': (0, 3),
        'thal': (0, 3)
    }
    return ranges

def get_risk_level(probability):
    """
    Determine risk level based on probability
    
    Returns: tuple (risk_level, color, message)
    """
    if probability < 0.3:
        return "Low Risk", "#28a745", "Low probability of heart disease"
    elif probability < 0.6:
        return "Moderate Risk", "#ffc107", "Moderate risk - Consider medical consultation"
    else:
        return "High Risk", "#dc3545", "High risk - Seek immediate medical attention"

def validate_input(input_data):
    """Validate input data against expected ranges"""
    ranges = get_feature_ranges()
    errors = []
    
    for feature, value in input_data.items():
        if feature in ranges:
            min_val, max_val = ranges[feature]
            if not (min_val <= value <= max_val):
                errors.append(f"{feature}: Value {value} is out of range [{min_val}, {max_val}]")
    
    return len(errors) == 0, errors

def format_input_summary(input_data):
    """Format input data for display"""
    descriptions = get_feature_descriptions()
    summary = []
    
    for feature, value in input_data.items():
        desc = descriptions.get(feature, feature)
        summary.append(f"**{desc}:** {value}")
    
    return "\n".join(summary)

def get_recommendations(prediction, probability):
    """Get health recommendations based on prediction"""
    if prediction == 1:  # Disease predicted
        recommendations = [
            "🏥 **Consult a cardiologist immediately** for proper diagnosis",
            "💊 Follow prescribed medications strictly",
            "🏃‍♂️ Adopt a heart-healthy lifestyle with regular exercise",
            "🥗 Follow a low-cholesterol, low-sodium diet",
            "🚭 Quit smoking and limit alcohol consumption",
            "😴 Ensure adequate sleep (7-8 hours daily)",
            "🧘‍♂️ Practice stress management techniques",
            "📊 Monitor blood pressure and cholesterol regularly"
        ]
    else:  # No disease predicted
        recommendations = [
            "✅ Maintain current healthy lifestyle habits",
            "🏃‍♂️ Continue regular physical activity (150 min/week)",
            "🥗 Follow a balanced, heart-healthy diet",
            "⚖️ Maintain healthy body weight",
            "📊 Get regular health check-ups annually",
            "😴 Ensure quality sleep (7-8 hours daily)",
            "🧘‍♂️ Manage stress through relaxation techniques",
            "🚭 Avoid smoking and excessive alcohol"
        ]
    
    return recommendations