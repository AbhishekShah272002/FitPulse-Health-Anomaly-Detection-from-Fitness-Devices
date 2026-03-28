🚀 FitPulse AI Platform

Health Anomaly Detection from Fitness Devices

📌 Overview

FitPulse AI Platform is an end-to-end data science application designed to analyze fitness device data and detect health anomalies using machine learning. The platform integrates data analytics, preprocessing, feature engineering, and predictive modeling into an interactive dashboard built with Streamlit.

🎯 Key Features

📊 Data Analytics Module

CSV data ingestion
Data quality checks (null values, column profiling)
Exploratory Data Analysis (EDA)

🧬 Machine Learning Pipeline

Feature extraction using TSFresh
Clustering: K-Means, DBSCAN
Dimensionality Reduction: PCA, t-SNE
Time Series Forecasting using Prophet

⚠️ Anomaly Detection

Detect unusual health patterns from fitness data

🖥️ Interactive Dashboard
Real-time insights and visualization
User-friendly interface

🏗️ Project Structure

FitPulse/
│
├── Fitpulse-App/           # Streamlit application
├── Documentation/          # Project docs & reports
├── Screenshots/            # UI screenshots
│
├── Milestone1.ipynb        # Data Analytics
├── Milestone2.ipynb        # Feature Engineering
├── Milestone3.ipynb        # ML Modeling
├── Milestone4.ipynb        # Insights_dashboard
│
├── preprocessing.py        # Data cleaning & preprocessing
├── pattern_extraction.py   # Feature extraction logic
├── anomaly_detection.py    # Anomaly detection models
├── main_app.py             # Main Streamlit app
│
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
└── LICENSE

⚙️ Tech Stack

Programming: Python
Libraries: Pandas, NumPy, Scikit-learn
ML Techniques: Clustering, Forecasting, Dimensionality Reduction
Visualization: Matplotlib, Plotly
Framework: Streamlit
Tools: Git, VS Code

🚀 How to Run the Project

1️⃣ Clone the Repository

git clone https://github.com/your-username/FitPulse-Health-Anomaly-Detection-from-Fitness-Devices.git
cd FitPulse-Health-Anomaly-Detection-from-Fitness-Devices
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Application
streamlit run main_app.py

📊 Use Case

This project helps in:

Monitoring fitness data 📈
Detecting abnormal health patterns ⚠️
Providing predictive insights for better health decisions 🧠

📸 Screenshots

https://github.com/AbhishekShah272002/FitPulse-Health-Anomaly-Detection-from-Fitness-Devices/tree/main/Screenshots

📈 Future Enhancements

Integration with real-time wearable devices
Deep learning-based anomaly detection
Deployment on cloud platforms
User authentication system
