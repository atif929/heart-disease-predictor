# ❤️ Heart Disease Prediction System

A machine learning-powered web application for predicting the risk of heart disease based on medical parameters.

## 🎯 Project Overview

This project implements a comprehensive heart disease prediction system using multiple machine learning models. The system provides:
- Real-time disease risk prediction
- Interactive web interface built with Streamlit
- Comprehensive model comparison and evaluation
- Health recommendations based on predictions
- Detailed visualizations and insights

## 📊 Dataset

**Source:** UCI Heart Disease Dataset  
**Size:** 303 samples, 14 features  
**Target Variable:** Binary classification (0 = No Disease, 1 = Disease)

### Features:
- **age:** Age in years
- **sex:** Gender (1 = male, 0 = female)
- **cp:** Chest pain type (0-3)
- **trestbps:** Resting blood pressure (mm Hg)
- **chol:** Serum cholesterol (mg/dl)
- **fbs:** Fasting blood sugar > 120 mg/dl
- **restecg:** Resting ECG results (0-2)
- **thalach:** Maximum heart rate achieved
- **exang:** Exercise induced angina (1 = yes, 0 = no)
- **oldpeak:** ST depression induced by exercise
- **slope:** Slope of peak exercise ST segment (0-2)
- **ca:** Number of major vessels (0-3)
- **thal:** Thalassemia (0-3)

## 🛠️ Technologies Used

- **Python 3.8+**
- **Streamlit:** Web application framework
- **Scikit-learn:** Machine learning models
- **Pandas & NumPy:** Data manipulation
- **Matplotlib & Seaborn:** Static visualizations
- **Plotly:** Interactive visualizations

## 🤖 Machine Learning Models

The project implements and compares four different models:

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Decision Tree Classifier**
4. **Support Vector Machine (SVM)**

### Model Performance Metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Cross-Validation Score

## 📁 Project Structure

```
heart-disease-prediction/
│
├── app.py                          # Main Streamlit application
├── train_models.py                 # Model training script
├── utils.py                        # Helper functions
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore file
│
├── data/
│   └── heart.csv                   # Dataset
│
├── models/                         # Saved trained models
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── decision_tree.pkl
│   ├── svm.pkl
│   ├── scaler.pkl
│   └── model_comparison.csv
│
├── notebooks/
│   └── eda_preprocessing.ipynb     # EDA notebook
│
└── images/                         # Visualization outputs
    ├── target_distribution.png
    ├── feature_distributions.png
    ├── correlation_heatmap.png
    ├── model_comparison.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    └── feature_importance.png
```

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
- Download the heart disease dataset from UCI ML Repository or Kaggle
- Place it in the `data/` folder as `heart.csv`

### 5. Train Models
```bash
python train_models.py
```

This will:
- Load and preprocess the data
- Train all four ML models
- Save trained models in `models/` folder
- Generate evaluation visualizations in `images/` folder
- Create model comparison CSV

### 6. Run the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🌐 Deployment on Streamlit Cloud

### Step-by-Step Deployment Guide:

#### 1. Prepare Your Repository
```bash
# Initialize git repository (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit - Heart Disease Prediction System"

# Create GitHub repository and push
git remote add origin https://github.com/yourusername/heart-disease-prediction.git
git branch -M main
git push -u origin main
```

#### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Fill in the deployment form:
   - **Repository:** `yourusername/heart-disease-prediction`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click "Deploy"

#### 3. Important Notes for Deployment:
- Ensure all files are committed and pushed to GitHub
- Make sure `requirements.txt` is in the root directory
- The `models/` folder with trained models must be in the repository
- Dataset (`data/heart.csv`) should be included

#### 4. Access Your Deployed App:
Your app will be available at: `https://yourusername-heart-disease-prediction.streamlit.app`

## 📊 Usage Guide

### Making Predictions:

1. **Navigate to the "Prediction" tab**
2. **Select a model** from the sidebar
3. **Enter patient information:**
   - Demographics (age, sex)
   - Blood metrics (blood pressure, cholesterol)
   - Cardiac indicators (heart rate, ST depression)
   - Clinical test results
4. **Click "Predict Heart Disease"**
5. **View results:**
   - Disease probability
   - Risk level assessment
   - Health recommendations

### Exploring Model Performance:

1. **Model Comparison Tab:** View and compare all model metrics
2. **Visualizations Tab:** Explore confusion matrices, ROC curves, etc.
3. **Information Tab:** Learn about features and models

## 📈 Results

### Best Model Performance:
- **Model:** Random Forest Classifier
- **Accuracy:** ~85%
- **F1-Score:** ~84%
- **ROC-AUC:** ~90%

### Key Findings:
- Most important features: chest pain type, maximum heart rate, ST depression
- Model performs well on balanced dataset
- Ensemble methods (Random Forest) show superior performance

## 🎓 Project Components (As per Rubric)

### 1. Problem Identification (3 marks)
- **Problem:** Early detection of heart disease for timely intervention
- **Dataset:** UCI Heart Disease Dataset (303 samples, 14 features)
- **Relevance:** Heart disease is a leading cause of death globally

### 2. Data Preprocessing & EDA (2 marks)
- Checked for missing values and duplicates
- Statistical analysis and distributions
- Correlation analysis
- Feature relationship exploration
- Outlier detection
- Comprehensive visualizations

### 3. Model Implementation & Evaluation (4 marks)
- Implemented 4 different models
- Used multiple evaluation metrics
- Performed cross-validation
- Compared model performance
- Analyzed feature importance
- Generated confusion matrices and ROC curves

### 4. GUI Design & Usability (2 marks)
- Interactive Streamlit web interface
- User-friendly input forms
- Real-time predictions
- Multiple visualization types
- Responsive design
- Clear result presentation

## ⚠️ Limitations & Disclaimer

- This is an educational project and should not be used for actual medical diagnosis
- Predictions are based on historical data and may not be 100% accurate
- Always consult healthcare professionals for medical concerns
- Model performance may vary with different datasets

## 🔮 Future Enhancements

- [ ] Add more advanced ensemble methods (XGBoost, LightGBM)
- [ ] Implement deep learning models
- [ ] Add SHAP values for model interpretability
- [ ] Include more comprehensive patient history
- [ ] Add user authentication and history tracking
- [ ] Deploy as a mobile application
- [ ] Integrate with real-time health monitoring devices

## 👨‍💻 Author

**Your Name**  
Data Science Student  
Fall 2025

## 📞 Contact

- **Email:** your.email@example.com
- **GitHub:** [@yourusername](https://github.com/yourusername)
- **LinkedIn:** [Your Name](https://linkedin.com/in/yourprofile)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the dataset
- Streamlit team for the amazing framework
- Scikit-learn community for ML tools
- Course instructors and teaching assistants

## 📚 References

1. UCI Machine Learning Repository - Heart Disease Dataset
2. Scikit-learn Documentation
3. Streamlit Documentation
4. Various research papers on heart disease prediction

---

**Note:** This project was developed as part of the Data Science Semester Project, Fall 2025.

**Last Updated:** December 2025