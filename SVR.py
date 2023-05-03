# Import các thư viện cần thiết
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Đọc dữ liệu từ tệp CSV và chuyển đổi các giá trị ngày/tháng/năm thành dạng số
df = pd.read_csv('Dữ liệu Lịch sử SSI.csv', parse_dates=['Ngày'], date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y'), thousands=',', decimal='.')

# Chuyển đổi kiểu dữ liệu của các cột "Ngày", "Lần cuối", "KL" sang số
df['Ngày'] = pd.to_datetime(df['Ngày'], errors='coerce')
df['Ngày'] = df['Ngày'].apply(lambda x: x.timestamp())
df['Ngày'] = df['Ngày'].astype('float')

# Loại bỏ ký tự "M" khỏi cột "KL" và chuyển đổi kiểu dữ liệu của cột "KL" sang số
df['KL'] = df['KL'].str.replace('M', '')
# Loại bỏ ký tự "K" khỏi cột "KL" và chuyển đổi kiểu dữ liệu của cột "KL" sang số

df['KL'] = df['KL'].str.replace('K', '')
df['KL'] = df['KL'].astype(float)

# Loại bỏ ký tự "%" khỏi cột "% Thay đổi" và chuyển đổi kiểu dữ liệu của cột "% Thay đổi" sang số
df['% Thay đổi'] = df['% Thay đổi'].str.replace('%', '')
df['% Thay đổi'] = df['% Thay đổi'].astype(float) / 100

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Tiền xử lý dữ liệu bằng cách chuẩn hóa các biến độc lập
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Xây dựng mô hình SVR trên tập huấn luyện
regressor = SVR(kernel='rbf')
regressor.fit(X_train, y_train)

# Đánh giá hiệu suất của mô hình trên tập kiểm tra
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
# mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
# mape = np.mean(np.abs((y_test - y_pred) / (y_test + 0.000001))) * 100
mask = y_test != 0
mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
rmse = np.sqrt(mse)


# In ra các chỉ số đánh giá
print("MSE: ", mse)
print("MAE: ", mae)
print("MAPE: ", mape)
print("RMSE: ", rmse)

import matplotlib.pyplot as plt

# Vẽ biểu đồ giá trị thực tế và giá trị dự báo
plt.plot(y_test, color='blue', label='Giá trị thực tế')
plt.plot(y_pred, color='red', label='Giá trị dự báo')
plt.title('Dự báo giá trị của mô hình SVR')
plt.xlabel('Mẫu')
plt.ylabel('Giá trị')
plt.legend()
plt.show()