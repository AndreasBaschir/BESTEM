import pandas as pd
import matplotlib.pyplot as plt

#Load data
data = pd.read_csv('./assets/Stocks.csv', low_memory=False)

#Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

#Filter out products that were sold only on one day in a year or sold few in total
data = data.groupby('Product_ID').filter(lambda x: len(x) > 1 and x['Sales'].sum() > 10)

#Detect anomalies
def detect_anomalies(data):
    data['prev_stock'] = data['EndOfDayStock'].shift(1)
    stock_anomalies = data[(data['EndOfDayStock']        == data['prev_stock']) | (data['EndOfDayStock'] < data['Sales'])].copy()  # Use .copy() to avoid SettingWithCopyWarning
    return stock_anomalies

anomalies = detect_anomalies(data)

#Write anomalies to a CSV file
anomalies.to_csv('anomalii.csv', index=False)

#Create another CSV file with the data without anomalies
clean_data = data.drop(anomalies.index)
clean_data = clean_data.drop(columns=['prev_stock'])
clean_data.to_csv('clean_data.csv', index=False)

#Convert columns to numeric if they are not already
anomalies['EndOfDayStock'] = pd.to_numeric(anomalies['EndOfDayStock'])
anomalies['prev_stock'] = pd.to_numeric(anomalies['prev_stock'])

#Calculate the fluctuation of the stock (absolute difference between consecutive EndOfDayStock values)
anomalies['StockFluctuation'] = abs(anomalies['EndOfDayStock'] - anomalies['prev_stock'])

#Select the top 10 anomalies based on stock fluctuation
top_10_anomalies = anomalies.nlargest(10, 'StockFluctuation')

#Write top 10 anomalies to a CSV file
top_10_anomalies.to_csv('top_10_anomalies.csv', index=False)

#Create a bar chart for the top 10 anomalies
plt.bar(top_10_anomalies['Product_ID'], top_10_anomalies['StockFluctuation'])
plt.xlabel('Product_ID')
plt.ylabel('Stock Fluctuation')
plt.title('Top 10 Anomalies Based on Stock Fluctuation')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("C:\\Users\\Andreas\\Desktop\\Best\\assets\\images\\sales.png")
plt.show()