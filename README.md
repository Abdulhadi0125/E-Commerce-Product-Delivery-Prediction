# E-Commerce Product Delivery Prediction

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Project-blue)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-green)

## 📌 Project Overview
This project aims to **predict whether an e-commerce product will be delivered on time or delayed** using machine learning. By analyzing various factors such as **product weight, discount offered, shipment mode, and customer behavior**, we can provide valuable insights to optimize delivery logistics.

## 🚀 Key Features
- **Exploratory Data Analysis (EDA)** to identify key factors affecting delivery time.
- **Feature engineering and data preprocessing** for model training.
- **Comparison of multiple ML models** (Logistic Regression, Decision Tree, Random Forest, XGBoost).
- **XGBoost selected for best performance (73.31% accuracy).**
- **Model deployment using Streamlit** for real-time predictions.

## 📊 Dataset Overview
- **Total Records:** 10,999
- **Total Features:** 12
- **Target Variable:** `Reached.on.Time_Y.N` (1 = On-time, 0 = Delayed)

### 📌 Key Features:
| Feature | Description |
|---------|-------------|
| `Warehouse_block` | Location of shipment (A to F) |
| `Mode_of_Shipment` | Transport method (Ship, Flight, Road) |
| `Discount_offered` | Promotional discount applied |
| `Product_importance` | Product priority level (Low, Medium, High) |
| `Weight_in_gms` | Product weight in grams |
| `Customer_care_calls` | Number of support calls by the customer |

## 🔍 Exploratory Data Analysis (EDA) Insights
- **Higher discounts are correlated with late deliveries** (possible stock issues or shipping delays).
- **Heavier products tend to have higher late delivery rates**.
- **Shipments via road have more delays compared to flights.**
- **Products from Warehouse F have the most shipments (likely near a seaport).**

## 🛠️ Data Preprocessing
- **Handled missing values** (Checked using `df.isnull().sum()` and found none).
- **Encoding categorical variables** (Used Label Encoding for categorical features).
- **Feature scaling** (Applied Standardization for weight and cost features).
- **Train-test split:** 80% training, 20% testing.

## 🤖 Machine Learning Models Used
| Model | Accuracy |
|------------|-----------|
| Logistic Regression | 65.23% |
| Decision Tree | 70.31% |
| Random Forest | 72.89% |
| **XGBoost** | **73.31%** ✅ |

**Final Model: XGBoost** was selected due to its highest accuracy and best AUC score.

## 📡 Model Deployment (Streamlit App)
- **Built a Streamlit web app** to take user inputs and predict delivery status.
- **Trained XGBoost model saved as `best_model.pkl`.**
- **User-friendly UI** for real-time predictions.

### 📷 Streamlit App Demo
![Streamlit App](https://your-app-demo-screenshot-url)

## 🏆 Key Business Recommendations
- Optimize **logistics for heavy shipments** to reduce delays.
- Improve **customer support for high-value orders** (₹200-250 orders have the most complaints).
- **Refine discount strategies**, as high discounts don’t necessarily improve sales.

## 📂 Project Structure
```
📁 E-Commerce-Delivery-Prediction/
│-- data/
│   │-- e_commerce_dataset.csv
│-- notebooks/
│   │-- eda.ipynb
│   │-- model_training.ipynb
│-- app/
│   │-- app.py  # Streamlit Deployment
│-- models/
│   │-- best_model.pkl
│   │-- scaler.pkl
│-- README.md
│-- requirements.txt
```

## 🛠 Installation & Usage
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/Abdulhadi0125/E-Commerce-Delivery-Prediction.git
cd E-Commerce-Delivery-Prediction
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit App
```sh
streamlit run app/app.py
```

## 📜 Requirements
```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
streamlit
```

## 📄 Certifications & Recognition
✅ **Certified by Boston Institute of Analytics** upon evaluation.

## 📬 Connect with Me
🔗 **LinkedIn:** [linkedin.com/in/abdulhadi](https://linkedin.com/in/abdulhadi)  
🔗 **GitHub:** [github.com/Abdulhadi0125](https://github.com/Abdulhadi0125)  
📩 **Email:** abdulhadi12b102@gmail.com

---
🚀 **Feel free to star ⭐ this repository if you found it helpful!**
