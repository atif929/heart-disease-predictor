"""
Heart Disease Prediction System
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import (load_models, load_model_comparison, predict_disease,
                   get_feature_descriptions, get_feature_ranges,
                   get_risk_level, get_recommendations, format_input_summary)
import os

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    h1 {
        color: #FF6B6B;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# Load models
@st.cache_resource
def load_app_models():
    return load_models()

@st.cache_data
def load_comparison():
    return load_model_comparison()

# Title and description
st.title("❤️ Heart Disease Prediction System")
st.markdown("""
<div style='text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem; margin-bottom: 2rem;'>
    <h3>AI-Powered Cardiovascular Risk Assessment</h3>
    <p>Enter patient data to predict the risk of heart disease using machine learning</p>
</div>
""", unsafe_allow_html=True)

# Load models and data
try:
    models, scaler = load_app_models()
    comparison_df = load_comparison()
    
    if not models:
        st.error("⚠️ Models not found! Please run 'train_models.py' first.")
        st.stop()
        
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Sidebar - Model Selection and Info
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    selected_model = st.selectbox(
        "Select Prediction Model",
        options=list(models.keys()),
        help="Choose the machine learning model for prediction"
    )
    
    st.markdown("---")
    
    # Model Performance
    st.subheader("📊 Model Performance")
    if comparison_df is not None:
        model_stats = comparison_df[comparison_df['Model'] == selected_model].iloc[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{model_stats['Accuracy']:.2%}")
            st.metric("Precision", f"{model_stats['Precision']:.2%}")
        with col2:
            st.metric("Recall", f"{model_stats['Recall']:.2%}")
            st.metric("F1-Score", f"{model_stats['F1-Score']:.2%}")
    
    st.markdown("---")
    
    # About section
    with st.expander("ℹ️ About This App"):
        st.markdown("""
        **Heart Disease Prediction System**
        
        This application uses machine learning to predict the likelihood of heart disease based on medical parameters.
        
        **Features:**
        - Multiple ML models
        - Real-time predictions
        - Risk assessment
        - Health recommendations
        
        **Note:** This is a prediction tool and should not replace professional medical advice.
        """)

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏥 Prediction", "📈 Model Comparison", "📊 Visualizations", "ℹ️ Information"])

# TAB 1: PREDICTION
with tab1:
    st.header("Patient Information Input")
    
    # Get feature ranges
    ranges = get_feature_ranges()
    
    # Create input form
    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics")
            age = st.slider("Age", int(ranges['age'][0]), int(ranges['age'][1]), 50)
            sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
            
            st.subheader("Blood Metrics")
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", 
                                int(ranges['trestbps'][0]), int(ranges['trestbps'][1]), 120)
            chol = st.slider("Cholesterol (mg/dl)", 
                            int(ranges['chol'][0]), int(ranges['chol'][1]), 200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", 
                              options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
        with col2:
            st.subheader("Cardiac Indicators")
            thalach = st.slider("Max Heart Rate Achieved", 
                               int(ranges['thalach'][0]), int(ranges['thalach'][1]), 150)
            oldpeak = st.slider("ST Depression", 
                               float(ranges['oldpeak'][0]), float(ranges['oldpeak'][1]), 1.0, 0.1)
            
            st.subheader("Clinical Tests")
            cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                             format_func=lambda x: ["Typical Angina", "Atypical Angina", 
                                                   "Non-anginal Pain", "Asymptomatic"][x])
            restecg = st.selectbox("Resting ECG", options=[0, 1, 2],
                                  format_func=lambda x: ["Normal", "ST-T Wave Abnormality", 
                                                        "Left Ventricular Hypertrophy"][x])
        
        with col3:
            st.subheader("Exercise & Other")
            exang = st.selectbox("Exercise Induced Angina", 
                                options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            slope = st.selectbox("Slope of Peak Exercise ST", options=[0, 1, 2],
                                format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
            ca = st.selectbox("Number of Major Vessels", options=[0, 1, 2, 3])
            thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3],
                               format_func=lambda x: ["Normal", "Fixed Defect", 
                                                     "Reversible Defect", "Unknown"][x])
        
        # Submit button
        submitted = st.form_submit_button("🔍 Predict Heart Disease", use_container_width=True)
    
    # Make prediction when form is submitted
    if submitted:
        # Collect input data
        input_data = {
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
            'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
            'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }
        
        # Make prediction
        try:
            model = models[selected_model]
            prediction, probability = predict_disease(model, scaler, input_data)
            
            # Store in session state
            st.session_state.prediction_made = True
            st.session_state.prediction = prediction
            st.session_state.probability = probability
            st.session_state.input_data = input_data
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Display results if prediction was made
    if st.session_state.prediction_made:
        st.markdown("---")
        st.header("🎯 Prediction Results")
        
        prediction = st.session_state.prediction
        probability = st.session_state.probability
        
        # Result display
        col1, col2, col3 = st.columns([2, 3, 2])
        
        with col2:
            # Disease probability
            disease_prob = probability[1]
            no_disease_prob = probability[0]
            risk_level, risk_color, risk_message = get_risk_level(disease_prob)
            
            # Result box
            if prediction == 1:
                st.markdown(f"""
                <div style='background-color: {risk_color}; padding: 2rem; border-radius: 1rem; text-align: center;'>
                    <h2 style='color: white; margin: 0;'>⚠️ Heart Disease Detected</h2>
                    <h1 style='color: white; font-size: 3rem; margin: 1rem 0;'>{disease_prob:.1%}</h1>
                    <p style='color: white; font-size: 1.2rem; margin: 0;'>{risk_message}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color: #28a745; padding: 2rem; border-radius: 1rem; text-align: center;'>
                    <h2 style='color: white; margin: 0;'>✅ No Heart Disease Detected</h2>
                    <h1 style='color: white; font-size: 3rem; margin: 1rem 0;'>{no_disease_prob:.1%}</h1>
                    <p style='color: white; font-size: 1.2rem; margin: 0;'>Low risk of heart disease</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Probability gauge chart
        st.markdown("### Probability Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gauge chart for disease probability
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=disease_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Disease Probability", 'font': {'size': 20}},
                number={'suffix': "%", 'font': {'size': 40}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': risk_color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#d4edda'},
                        {'range': [30, 60], 'color': '#fff3cd'},
                        {'range': [60, 100], 'color': '#f8d7da'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 60
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            # Probability bar chart
            fig_bar = go.Figure(data=[
                go.Bar(name='Probability', x=['No Disease', 'Disease'], 
                      y=[no_disease_prob, disease_prob],
                      marker_color=['#28a745', risk_color],
                      text=[f'{no_disease_prob:.1%}', f'{disease_prob:.1%}'],
                      textposition='auto')
            ])
            fig_bar.update_layout(
                title="Prediction Probabilities",
                yaxis_title="Probability",
                yaxis_range=[0, 1],
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Recommendations
        st.markdown("---")
        st.header("💡 Health Recommendations")
        
        recommendations = get_recommendations(prediction, disease_prob)
        
        cols = st.columns(2)
        for idx, rec in enumerate(recommendations):
            with cols[idx % 2]:
                st.markdown(f"- {rec}")
        
        # Input summary
        st.markdown("---")
        with st.expander("📋 View Input Summary"):
            input_summary = format_input_summary(st.session_state.input_data)
            st.markdown(input_summary)

# TAB 2: MODEL COMPARISON
with tab2:
    st.header("📈 Model Performance Comparison")
    
    if comparison_df is not None:
        # Display comparison table
        st.subheader("Performance Metrics Table")
        st.dataframe(
            comparison_df.style.format({
                'Accuracy': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1-Score': '{:.4f}',
                'ROC-AUC': '{:.4f}',
                'CV Score': '{:.4f}'
            }).background_gradient(cmap='RdYlGn', subset=['Accuracy', 'F1-Score']),
            use_container_width=True
        )
        
        # Metrics comparison charts
        st.subheader("Visual Comparison")
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart for all metrics
            fig_metrics = go.Figure()
            
            for metric in metrics:
                fig_metrics.add_trace(go.Bar(
                    name=metric,
                    x=comparison_df['Model'],
                    y=comparison_df[metric],
                    text=comparison_df[metric].apply(lambda x: f'{x:.3f}'),
                    textposition='auto'
                ))
            
            fig_metrics.update_layout(
                title="All Metrics Comparison",
                xaxis_title="Model",
                yaxis_title="Score",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        with col2:
            # Radar chart
            fig_radar = go.Figure()
            
            for idx, row in comparison_df.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row['Accuracy'], row['Precision'], row['Recall'], 
                       row['F1-Score'], row['ROC-AUC']],
                    theta=metrics,
                    fill='toself',
                    name=row['Model']
                ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Model Performance Radar Chart",
                height=400
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Best model highlight
        best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
        st.success(f"""
        🏆 **Best Performing Model:** {best_model['Model']}  
        📊 **Accuracy:** {best_model['Accuracy']:.4f} | **F1-Score:** {best_model['F1-Score']:.4f}
        """)

# TAB 3: VISUALIZATIONS
with tab3:
    st.header("📊 Data Visualizations")
    
    viz_option = st.selectbox(
        "Select Visualization",
        ["Model Comparison", "Confusion Matrices", "ROC Curves", "Feature Importance"]
    )
    
    # Map options to image files
    viz_images = {
        "Model Comparison": "images/model_comparison.png",
        "Confusion Matrices": "images/confusion_matrices.png",
        "ROC Curves": "images/roc_curves.png",
        "Feature Importance": "images/feature_importance.png"
    }
    
    img_path = viz_images[viz_option]
    
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    else:
        st.warning("⚠️ Visualization not available. Please run 'train_models.py' first.")

# TAB 4: INFORMATION
with tab4:
    st.header("ℹ️ About Heart Disease Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Feature Descriptions")
        descriptions = get_feature_descriptions()
        
        for feature, desc in descriptions.items():
            with st.expander(f"**{feature.upper()}**"):
                st.write(desc)
                ranges = get_feature_ranges()
                if feature in ranges:
                    st.info(f"Valid Range: {ranges[feature][0]} - {ranges[feature][1]}")
    
    with col2:
        st.subheader("🎯 About the Models")
        
        st.markdown("""
        #### Logistic Regression
        A statistical model that predicts binary outcomes using a logistic function.
        - **Pros:** Simple, interpretable, fast
        - **Cons:** Assumes linear relationships
        
        #### Random Forest
        An ensemble method using multiple decision trees.
        - **Pros:** High accuracy, handles non-linearity
        - **Cons:** Less interpretable, computationally intensive
        
        #### Decision Tree
        A tree-like model of decisions and outcomes.
        - **Pros:** Easy to understand, handles non-linearity
        - **Cons:** Prone to overfitting
        
        #### Support Vector Machine (SVM)
        Finds optimal hyperplane to separate classes.
        - **Pros:** Effective in high dimensions
        - **Cons:** Slow with large datasets
        """)
        
        st.subheader("⚠️ Disclaimer")
        st.warning("""
        **Important Notice:**
        
        This application is for educational and informational purposes only. 
        It should not be used as a substitute for professional medical advice, 
        diagnosis, or treatment. Always consult with qualified healthcare 
        providers for medical concerns.
        
        The predictions are based on machine learning models trained on 
        historical data and may not be 100% accurate.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 1rem; color: #666;'>
    <p>❤️ Heart Disease Prediction System | Data Science Semester Project Fall 2025</p>
    <p>Developed with Streamlit, Scikit-learn, and Plotly</p>
</div>
""", unsafe_allow_html=True)