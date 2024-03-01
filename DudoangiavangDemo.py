import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Đọc dữ liệu từ tệp Excel
data = pd.read_excel('GoldPriceData.xlsx')

# Cập nhật định dạng ngày tháng để phù hợp với dữ liệu
data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')

# Tạo cột mới cho ngày trong năm
data['DayOfYear'] = data['Date'].dt.dayofyear

# Chọn biến độc lập và biến phụ thuộc 
X = data[['DayOfYear', 'interest_rate', 'cpi', 'pmi', 'unemployment', 'usd_price']]
y = data['Open']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=100)

# Chuẩn hóa các đặc trưng (standardization) với tên đặc trưng
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Tạo mô hình hồi quy tuyến tính đa biến
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán giá vàng dựa trên dữ liệu kiểm tra
y_pred = model.predict(X_test)

# Tạo một danh sách trống để lưu các dự đoán
predicted_prices = []

# Dự đoán giá vàng cho mỗi ngày trong tương lai
for i in range(5):
    datestart = '5'
    stringDate = datestart + '-Oct-23'
    # Tạo ngày mới cho dự đoán
    new_date = datetime.strptime(stringDate, '%d-%b-%y')
    new_date += pd.DateOffset(days=i)  # Cập nhật ngày cho mỗi lần lặp
    new_day_of_year = new_date.timetuple().tm_yday
    new_data = [[new_day_of_year, 5.5, 3.7, 48, 3.6, 24000]]
    new_data = scaler.transform(new_data)  
    predicted_price = model.predict(new_data)
    predicted_prices.append(predicted_price[0])
    # In dự đoán cho mỗi ngày
    print(f'Giá vàng dự đoán cho ngày {new_date}: {predicted_prices[i]}')

# Tạo biểu đồ
plt.figure(figsize=(15, 15))
plt.plot(data['Date'][:len(y_test)], y_test, label='Giá vàng thực tế')
plt.plot(data['Date'][:len(y_test)], y_pred, label='Giá vàng dự đoán')
plt.axvline(x=new_date, color='r', linestyle='--', label='Ngày dự đoán')
plt.xlabel('Ngày')
plt.ylabel('Giá vàng')
plt.title('Dự đoán giá vàng')
plt.legend()
plt.show()
