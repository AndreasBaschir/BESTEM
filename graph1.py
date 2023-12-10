import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as fig
import matplotlib as mat

data = pd.read_csv('Stocks.csv',low_memory=False)
data['Date'] = pd.to_datetime(data['Date'])
target_product_id_1 = str(21092)  

product_data_1 = data[data['Product_ID'] == target_product_id_1]

plt.figure(figsize=(12, 8))
plt.plot(product_data_1['Date'], product_data_1['Sales'])
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title(f"Sales across time for #{target_product_id_1}")
plt.grid(True)
plt.tight_layout()
plt.savefig('graph1.png')

target_product_id_2 = str(23166)  

product_data_2 = data[data['Product_ID'] == target_product_id_2]

plt.figure(figsize=(12, 8))
plt.plot(product_data_2['Date'], product_data_2['Sales'])
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title(f"Sales across time for #{target_product_id_2}")
plt.grid(True)
plt.tight_layout()
plt.savefig('graph2.png')

target_product_id_3 = str(35961)  

product_data_3 = data[data['Product_ID'] == target_product_id_3]

plt.figure(figsize=(12, 8))
plt.plot(product_data_3['Date'], product_data_3['Sales'])
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title(f"Sales across time for #{target_product_id_3}")
plt.grid(True)
plt.tight_layout()
plt.savefig('graph3.png')

target_product_id_4 = str(20759)  

product_data_4 = data[data['Product_ID'] == target_product_id_4]

plt.figure(figsize=(12, 8))
plt.plot(product_data_4['Date'], product_data_4['Sales'])
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title(f"Sales across time for #{target_product_id_4}")
plt.grid(True)
plt.tight_layout()
plt.savefig('graph4.png')

target_product_id_5 = str(16044)  

product_data_5 = data[data['Product_ID'] == target_product_id_5]

plt.figure(figsize=(12, 8))
plt.plot(product_data_5['Date'], product_data_5['Sales'])
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title(f"Sales across time for #{target_product_id_5}")
plt.grid(True)
plt.tight_layout()
plt.savefig('graph5.png')

target_product_id_6 = str(35955)  

product_data_6 = data[data['Product_ID'] == target_product_id_6]

plt.figure(figsize=(12, 8))
plt.plot(product_data_6['Date'], product_data_6['Sales'])
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title(f"Sales across time for #{target_product_id_6}")
plt.grid(True)
plt.tight_layout()
plt.savefig('graph6.png')

target_product_id_7 = str(16046)  

product_data_7 = data[data['Product_ID'] == target_product_id_7]

plt.figure(figsize=(12, 8))
plt.plot(product_data_7['Date'], product_data_7['Sales'])
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title(f"Sales across time for #{target_product_id_7}")
plt.grid(True)
plt.tight_layout()
plt.savefig('graph7.png')

target_product_id_8 = str(21697)  

product_data_8 = data[data['Product_ID'] == target_product_id_8]

plt.figure(figsize=(12, 8))
plt.plot(product_data_8['Date'], product_data_8['Sales'])
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title(f"Sales across time for #{target_product_id_8}")
plt.grid(True)
plt.tight_layout()
plt.savefig('graph8.png')

target_product_id_9 = str(21017)  

product_data_9 = data[data['Product_ID'] == target_product_id_9]

plt.figure(figsize=(12, 8))
plt.plot(product_data_9['Date'], product_data_9['Sales'])
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title(f"Sales across time for #{target_product_id_9}")
plt.grid(True)
plt.tight_layout()
plt.savefig('graph9.png')

target_product_id_10 = str(16046)  

product_data_10 = data[data['Product_ID'] == target_product_id_10]

plt.figure(figsize=(12, 8))
plt.plot(product_data_10['Date'], product_data_10['Sales'])
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title(f"Sales across time for #{target_product_id_10}")
plt.grid(True)
plt.tight_layout()
plt.savefig('graph10.png')