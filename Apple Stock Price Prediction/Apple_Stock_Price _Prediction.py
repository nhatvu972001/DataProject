import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.dates import  DateFormatter
from pyspark.sql.functions import  col,  to_date
import datetime as dt
import findspark
from pyspark.sql import SparkSession

findspark.init()

# Tạo SparkSession
spark = SparkSession.builder.getOrCreate()

# Tạo dataframe từ file AAPL.csv
df = spark.read.csv('AAPL.csv', header=True, inferSchema=True)

# Hiển thị DataFrame và các thông tin cần thiết khác về dữ liệu
df.show()
df.printSchema()
df.collect()

# Thêm cột Avg_price vào DataFrame
df = df.withColumn("Avg_price", (col("Open") + col("Close")) / 2)
print("=================================================================")
print("DataFrame sau khi đã được thêm cột Avg_price ")
df.show()

# Hiển thị các thông số mô tả bộ dữ liệu
print("=================================================================")
print("Các thông số mô tả bộ dữ liệu")
df.describe().show()

# In ra 2 cột chính Date và Avg_price
print("=================================================================")
print("DataFrame gồm 2 cột Date và Avg_price")
df_avg_price = df.select('Date', 'Avg_price')
df_avg_price.show()

# Chuyển đổi định dạng ngày tháng
df_avg_price = df_avg_price.withColumn('Date', to_date(df['Date'], 'dd/MM/yyyy'))
# Chuyển đổi DataFrame thành một mảng NumPy
arr = df_avg_price.toPandas().values

# Tách cột Date và cột Avg_price thành 2 mảng riêng
dates = arr[:, 0]
avg_prices = arr[:, 1]

# Tạo biểu đồ đường
plt.plot(dates, avg_prices)

# Đặt tên cho các trục và hiển thị biểu đồ
plt.xlabel("Years(Time)")
plt.ylabel("Avg Price")
print("=================================================================")
print("Biểu đồ giá theo thời gian")
plt.title("Chart showing Apple stock price over time")
plt.show()

# Chuyển đổi định dạng ngày tháng
filtered_df = df.withColumn('Date', to_date(col('Date'), 'yyyy-MM-dd'))

# Lọc dữ liệu
filtered_df = df.filter((to_date(col("Date"), "dd/MM/yyyy") >= "1990-01-02") & (to_date(col("Date"), "dd/MM/yyyy") <= "2009-01-02"))
num_rows = filtered_df.count()
print("=================================================================")
print("Số dòng từ ngày 2/1/1990 đến ngày 2/1/2009 là:", num_rows)

# Đổi định dạng cột Date sang ngày tháng năm
df = df.withColumn('Date', to_date(df['Date'], 'dd/MM/yyyy'))

# Chia bộ dữ liệu ra thành 2 trước và sau ngày 2/1/2009
filtered_df = df.filter(col('Date') >= '2009-01-02')
new_df = filtered_df.select('Date', 'Avg_price')
print("=================================================================")
print("DataFrame từ ngày 2/1/2009 trở về sau")
new_df.show()

# Chuyển đổi DataFrame thành một mảng NumPy
arr = new_df.toPandas().values

# Tách cột Date và cột Avg_price thành 2 mảng riêng
dates = arr[:, 0]
avg_prices = arr[:, 1]

# Tạo biểu đồ đường
plt.plot(dates, avg_prices)

# Đặt tên cho các trục và hiển thị biểu đồ
plt.xlabel("Years(Time)")
plt.ylabel("Avg Price")
print("=================================================================")
print("Biểu đồ giá theo thời gian theo DataFrame mới")
plt.title("Chart showing Apple stock price over time")
plt.show()

# Chia dữ DataFrame thành 2 tệp train  và test 
train_data = new_df.filter(col("Date") < "2021-05-30")
test_data = new_df.filter(col("Date") >= "2021-05-30")

# Chuyển tụi này sang Pandas Data Frame
train_data = train_data.toPandas()
test_data_pd = test_data.toPandas()
new_df = new_df.toPandas()


# Chuẩn hóa dữ liệu bằng MinMaxScaler
scaler = MinMaxScaler()
train_data = train_data['Avg_price'].values.reshape(-1,1)
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data_pd = scaler.fit_transform(test_data_pd[['Avg_price']])

# Tạo model
model = Sequential()
n_features = 1 # số lượng đặc trưng đầu vào
n_input = 126

# Định nghĩa mô hình LSTM
model.add(LSTM(units=100, return_sequences=True, input_shape=(n_input, n_features)))
model.add(Dropout(0.3))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(25))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Sử dụng EarlyStopping để dừng huấn luyện nếu không có sự cải thiện về độ chính xác
early_stop = EarlyStopping(monitor='val_loss', patience=10)
print("=================================================================")
print("In ra các thông só của mô hình LSTM")
model.summary()

# Tạo TimeSeriesGenerator để phát sinh chuỗi thời gian
train_generator = TimeseriesGenerator(train_data, train_data, length=n_input, batch_size=32)

# Tạo generator cho dữ liệu kiểm tra
test_generator = TimeseriesGenerator(test_data_pd, test_data_pd, length=n_input, batch_size=32)
print("=================================================================")
print("Hiển thị quá trình huấn luyện mô hình")
# Huấn luyện mô hình
model.fit(train_generator, epochs=50, validation_data=(test_generator), callbacks=[early_stop])


# Hiển thị biểu đồ loss và value loss
print("=================================================================")
print("Hiển thị biểu đồ giá trị loss và val_loss")
loss = pd.DataFrame(model.history.history)
loss.plot()

# Dự đoán giá cổ phiếu của Apple
# Tạo mảng test_predictions
test_predictions = []
current_batch = train_data[-n_input:].reshape((1, n_input, n_features))

print("=================================================================")
print("Hiển thị quá trình thực thi mô hình")
# Sử dụng vòng lặp để dự đoán tất cả các giá trị có trong test_data_pd
for i in range(len(test_data_pd)):
    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred)
    current_batch = np.append(current_batch[:, 1:, :],[[current_pred]],axis=1)

# Chuyển data frame test_predictions về dạng ban đầu
test_predictions = scaler.inverse_transform(test_predictions)
test_data= test_data.toPandas()
print("=================================================================")
print("In ra Data Frame của test_data (cho vui)")
test_data.head()

# Tạo pandas DataFrame từ test_predictions
predictions_df = pd.DataFrame(test_predictions, columns=["Predictions"])

# Gộp predictions_df với test_data
test_data = pd.concat([test_data.iloc[0:len(predictions_df)], predictions_df], axis=1)
print("=================================================================")
print("In ra Data Frame của test_data sau khi thêm cột Predictions")
test_data.head()

# Vẽ biểu đồ
plt.plot(test_data['Avg_price'], label='Avg_price', color='blue')
plt.plot(test_data['Predictions'], label='Predictions', color='orange')

# Đặt các thông số cho biểu đồ
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Apple Stock Price Prediction')
plt.legend()

# Hiển thị biểu đồ
print("=================================================================")
print("Hiển thị biểu đồ giá thực tế và giá dự đoán")
plt.show()

from sklearn.metrics import mean_squared_error
print("=================================================================")
print("In ra độ đo MSE")
np.sqrt(mean_squared_error(test_data['Avg_price'],test_data['Predictions']))

from dateutil.parser import parse
from dateutil.rrule import rrule, DAILY, MO, TU, WE, TH, FR

# Chọn ra các ngày trong tuần từ thứ 2 đến thứ 6
result = rrule(
  DAILY,
  byweekday=(MO,TU,WE,TH,FR),
  dtstart=parse('2021-12-01'),
  until=parse('2022-05-31')
)
print("=================================================================")
print("In ra danh sách các ngày sẽ dự đoán tiếp theo")
list(result)

# Tạo mảng forecast_date
forecast_date = []

# Sử dụng vòng lặp để tạo trình thời gian chứa tất cả các số ngày thuộc thứ 2 đến thứ 6 của các ngày từ nàgy 12/1/2021 đến ngày 31/5/2022
for i in list(result):
    forecast_date.append(i.strftime('%Y-%m-%d'))
forecast_date_df = pd.DataFrame(forecast_date, columns=['Date'])
forecast_date_df['Date'] = pd.to_datetime(forecast_date_df['Date'], format='%Y-%m-%d')
print("=================================================================")
print("In ra số dòng sẽ dự đoán")
len(forecast_date_df)

forecast = []
# Dự đoán giá cổ phiếu trong nữa năm đầu 2022
periods = 128
test_data_fc = scaler.fit_transform(test_data[['Avg_price']])
first_eval_batch = test_data_fc[-periods:]
current_batch = first_eval_batch.reshape((1, periods, n_features))

print("=================================================================")
print("Hiển thị quá trình thực thi mô hình trên forecast_date_df ")

# Sử dụng vòng lặp để dự đoán tất cả các giá trị có trong predios
for i in range(periods):
    current_pred = model.predict(current_batch)[0]
    forecast.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

# Chuyển dữ liệu về dạng ban đầu
forecast = scaler.inverse_transform(forecast)

# Chuyển forecast thành pandas dataframe với tên cột Predictions
forecast_df = pd.DataFrame(forecast, columns=['Predictions'])

# Gộp forecast_date_df và forecast_df theo cột bằng hàm concat
forecast_data = pd.concat([forecast_date_df.iloc[0:len(forecast_df)], forecast_df], axis=1)
print("=================================================================")
print("In ra Data Frame của forecast_data")
forecast_data.head(130)

# Chuyển dataframe thành mảng numpy
forecast_data_array = forecast_data.values
# Trích xuất cột ngày và cột dự đoán
dates = forecast_data_array[:, 0]
predictions = forecast_data_array[:, 1]
plt.plot(dates, predictions)
plt.title('Predictions for 2022')
plt.xlabel('Date')
plt.ylabel('Price')
print("=================================================================")
print("Hiển thị biểu đồ giá dự đoán từ ngày 1/12/2021 đến ngày 31/5/2022")
plt.show()