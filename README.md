# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 
02.11.25


### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM:
~~~
# --- Import Libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# --- Load the dataset ---
data = pd.read_csv('housing_price_dataset.csv')

# --- Prepare yearly average prices ---
data_yearly = data.groupby('YearBuilt')['Price'].mean().sort_index()

# Convert YearBuilt to datetime (for time series analysis)
data_yearly.index = pd.to_datetime(data_yearly.index, format='%Y')

# --- Basic exploration ---
print(data_yearly.head())
plt.figure(figsize=(10, 4))
plt.plot(data_yearly, label='Average House Price by Year Built')
plt.xlabel('Year')
plt.ylabel('Average Price')
plt.title('Yearly Average House Prices')
plt.legend()
plt.grid(True)
plt.show()

# --- Perform Augmented Dickey-Fuller test (stationarity check) ---
result = adfuller(data_yearly)
print('ADF Statistic:', result[0])
print('p-value:', result[1])
if result[1] > 0.05:
    print("❗ The series is likely non-stationary — differencing may be needed.")
else:
    print("✅ The series appears stationary.")

# --- Split into training and testing sets ---
x = int(0.8 * len(data_yearly))
train_data = data_yearly.iloc[:x]
test_data = data_yearly.iloc[x:]

# --- Plot ACF and PACF to determine lag order ---
plt.figure(figsize=(10, 5))
plot_acf(data_yearly, lags=20, alpha=0.05)
plt.title('Autocorrelation Function (ACF) - House Prices')
plt.show()

plt.figure(figsize=(10, 5))
plot_pacf(data_yearly, lags=20, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF) - House Prices')
plt.show()

# --- Fit AutoRegressive (AR) model ---
lag_order = 3  # you can adjust after viewing PACF plot
model = AutoReg(train_data, lags=lag_order)
model_fit = model.fit()
print(model_fit.summary())

# --- Make predictions ---
predictions = model_fit.predict(
    start=len(train_data),
    end=len(train_data) + len(test_data) - 1,
    dynamic=False
)

# --- Evaluate model performance ---
mse = mean_squared_error(test_data, predictions)
print('Mean Squared Error (MSE):', mse)
print('Root Mean Squared Error (RMSE):', np.sqrt(mse))

# --- Plot actual vs predicted prices ---
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data.values, label='Actual House Prices')
plt.plot(test_data.index, predictions, label='Predicted Prices', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Average Price')
plt.title('AR Model Predictions vs Actual Test Data (Housing Prices)')
plt.legend()
plt.grid(True)
plt.show()
~~~
### OUTPUT:

GIVEN DATA
<img width="923" height="449" alt="506438835-641a54f1-eef8-4226-958d-e1f8697d7969" src="https://github.com/user-attachments/assets/159e66c1-2c80-4854-9713-caf7d04e9505" />

PACF - ACF

<img width="590" height="440" alt="506439022-3596f340-7479-4b8f-b72f-825f505a87d3" src="https://github.com/user-attachments/assets/3a43f174-18ae-4409-a8ca-e67a7cf04e28" />
<img width="569" height="443" alt="506439399-0b1654ad-231c-41af-a161-85744c6d18d8" src="https://github.com/user-attachments/assets/2dff4f6f-8eab-40e4-b72e-2128abdf437d" />


FINIAL PREDICTION
<img width="1090" height="620" alt="506439599-4250d6e5-001a-4e42-a84b-dfd377f59267" src="https://github.com/user-attachments/assets/288c013b-47ce-47f2-bc4e-53534401436b" />

### RESULT:
Thus we have successfully implemented the auto regression function using python.
