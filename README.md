# House-Price-Prediction-using-Python-and-ML

# House Price Prediction using Python and Machine Learning

## 📌 Overview
This project aims to predict house prices based on various features such as location, size, number of rooms, and other factors. Using Machine Learning techniques, we develop a predictive model that can estimate house prices accurately.

## 🛠️ Tech Stack
- Python
- Pandas, NumPy (Data Processing)
- Scikit-learn (Machine Learning)
- Matplotlib, Seaborn (Data Visualization)
- Jupyter Notebook
- Joblib (Model Persistence)

## 📂 Dataset
We use the **Boston Housing Dataset**, which contains **506 rows** and **14 features**, including:
- CRIM: Crime rate per capita
- ZN: Proportion of residential land zoned for large lots
- INDUS: Proportion of non-retail business acres per town
- CHAS: Charles River dummy variable (0 or 1)
- NOX: Nitric oxides concentration
- RM: Average number of rooms per dwelling
- AGE: Proportion of owner-occupied units built before 1940
- DIS: Weighted distance to employment centers
- RAD: Index of accessibility to radial highways
- TAX: Property tax rate per $10,000
- PTRATIO: Pupil-teacher ratio
- B: Proportion of Black residents
- LSTAT: Percentage of lower status population
- MEDV: Median house value (Target variable, in $1000s)

## 🔍 Exploratory Data Analysis (EDA)
- Data visualization using Matplotlib and Seaborn
- Handling missing values (RM column has 5 missing values)
- Correlation analysis to identify significant predictors

## 🏗️ Model Building
- Data preprocessing (feature scaling, encoding categorical variables)
- Splitting the dataset into training and testing sets
- Implementing multiple machine learning models:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - XGBoost
- Model evaluation using RMSE, MAE, and R² score

## 🏆 Trained Model
- The trained **Decision Tree Regressor** model is saved as `Dragon.joblib`.
- Warning: The model was trained with `scikit-learn 1.6.0`, which may cause compatibility issues with older versions.

## 📊 Results
- The best-performing model is selected based on evaluation metrics.
- Predictions are compared with actual values.

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Gaurav-Raj64/House-Price-Prediction-using-Python-and-ML.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook or Python script to train and test the model.

## 📌 Future Enhancements
- Hyperparameter tuning for better accuracy
- Deployment using Flask or Streamlit
- Adding more real-world features to improve predictions

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repo and submit a pull request.

## 📧 Contact
For any queries, reach out to me at: grajgaurav2022@gmail.com

