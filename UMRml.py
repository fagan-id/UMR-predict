# importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# reading the datasets about Gaji UMR Indonesia over the year
data = pd.read_csv('Indonesian Salary by Region (1997-2022).csv')
data = data.dropna()
data.rename(columns={'REGION': 'Region', 'SALARY': 'Salary', 'YEAR': 'Year'}, inplace=True)
data.Salary = pd.to_numeric(data.Salary, errors='coerce')
data

# defining the target for X = "Year", and y = "Salary"
X = data[['Year']]
y = data['Salary']

# splitting and then training the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ml_model = LinearRegression()

# training the model
ml_model.fit(X,y)

# prediction
y_pred = ml_model.predict(X_test)

#data visualisation using LinearRegression Models
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Year')
plt.ylabel('Salary (IDR)')
plt.title('Actual vs. Predicted Salaries')
plt.legend()
plt.grid()	
plt.show()

# prediksi rata-rata Gaji UMR untuk tahun-tahun depan
masa_depan = np.array([2023, 2024, 2025]).reshape(-1, 1)
prediksi_masa_depan = ml_model.predict(masa_depan)
print("Prediksi Gaji masa depan:")
for year, salary in zip(masa_depan.flatten(), prediksi_masa_depan):
    print(f"Year: {year}, Prediksi Gaji: Rp. {salary:.2f}")
