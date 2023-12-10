import numpy as np
from prophet import Prophet
from scipy import stats
import multiprocessing as mp
from scipy.stats import norm
from stockpyl.eoq import economic_order_quantity
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go



if __name__ == '__main__':
    data = pd.read_csv('Stocks.csv', low_memory=False)

    # Convert the 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Filter out products that were sold only on one day in a year or sold few in total
    data = data.groupby('Product_ID').filter(lambda x: len(x) > 1 and x['Sales'].sum() > 10)

    def detect_anomalies(data):
        data['prev_stock'] = data['EndOfDayStock'].shift(1)
        stock_anomalies = data[(data['EndOfDayStock'] == data['prev_stock']) | (data['EndOfDayStock'] < data['Sales'])]
        return stock_anomalies

    anomalies = detect_anomalies(data)

    #drop column prev_stock
    anomalies = anomalies.drop(columns=['prev_stock'])

    #create another csv file with the data without anomalies
    data = data.drop(anomalies.index)
    data = data.drop(columns=['prev_stock'])

    # Get the unique product IDs
    unique_product_ids = data['Product_ID'].unique()

    # Calculate the midpoint of the array
    point = len(unique_product_ids) // 46

    # Slice the array to include only the first half of the product IDs
    product_ids = unique_product_ids[:point]

    mape_median = 0
    count = 0

    # Initialize an empty dictionary to store the product IDs and their corresponding order quantities
    order_quantities = {}

    # Iterate over the unique product IDs
    for product_id in product_ids:
        # Do something with each product_id
        # print(product_id)
        old_filtered_data = data[data['Product_ID'] == product_id]
        initial_inventory = old_filtered_data['EndOfDayStock'].iloc[0] + old_filtered_data['Sales'].iloc[0]
        # # Assuming 'filtered_data' is your DataFrame and 'Sales' is the column where you want to remove outliers
        z_scores = stats.zscore(old_filtered_data['Sales'])
        filtered_data = old_filtered_data
        #
        if len(old_filtered_data) >= 100:
            for z in range(300):
                z = z * 0.01
                filtered_data = old_filtered_data[(z_scores < z) & (z_scores > -z)]
                if (len(filtered_data) >= 0.8 * len(old_filtered_data)):
                    break
        #
        # # train test split
        train_size = int(len(filtered_data) * 0.8)
        train, test = filtered_data[0:train_size], filtered_data[train_size:len(filtered_data)]
        #
        # # print(len(filtered_data))
        # # print(len(train))
        # # print(len(test))
        #
        # # Prophet requires the variable names in the time series to be:
        # # y – Target
        # # ds – Datetime
        train['ds'] = train.Date
        train['y'] = train.Sales
        train.drop(['Sales'], axis=1, inplace=True)
        #
        # # confidence interval
        model1 = Prophet(changepoint_prior_scale=0.05, interval_width=0.95, daily_seasonality=True)  # by default is 80%
        #
        # # Check if the DataFrame has at least two non-NaN rows
        if train['y'].count() < 2:
            # print(f'Skipping product_id {product_id} due to insufficient data')
            continue
        #
        model1.fit(train)
        #
        future = model1.make_future_dataframe(periods=365, freq='D')

        forecast = model1.predict(future)
        forecast_copy = forecast
        # print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

        # components = model1.plot_components(forecast)
        # components.show()

        # print(test[['Date', 'Sales']])

        # Convert 'Date' column to datetime in 'test'
        test['Date'] = pd.to_datetime(test['Date'])

        # Set 'Date' as the index in 'test'
        test.set_index('Date', inplace=True)

        # Filter 'forecast' to only include dates that are in 'test'
        forecast = forecast[forecast['ds'].isin(test.index)]

        # Calculate MAPE (Mean absolute percentage error)
        first_row_ds = forecast['ds'].iloc[0]
        forecast.set_index('ds', inplace=True)  # Uncomment this while running

        temp = (test['Sales'] - forecast.loc[first_row_ds:, 'yhat'])
        mape = (temp.abs() / test['Sales']).mean() * 100



        # print("MAPE: ", mape, "%")
        # mape_median += mape
        # count += 1
        # print(temp)
        #
        # print(forecast)
        # print(test)

        # Create a pandas Series with the predicted values and date indices
        forecasted_demand = pd.Series(forecast['yhat'].values, index=forecast.index)

        # Lead time (number of days it takes to replenish inventory)
        lead_time = 1  # it's different for every business, 1 is an example

        # Service level (probability of not stocking out)
        service_level = 0.95  # it's different for every business, 0.95 is an example

        # Calculate the optimal order quantity using the Newsvendor formula
        z = np.abs(np.percentile(forecasted_demand, 100 * (1 - service_level)))
        order_quantity = np.ceil(forecasted_demand.mean() + z).astype(int)

        if mape > 0 and mape < 100:
            # print(product_id)
            # print("MAPE: ", mape, "%")
            # Add the product ID and its order quantity to the dictionary
            order_quantities[product_id] = order_quantity

        # Calculate the reorder point
        reorder_point = round(forecasted_demand.mean() * lead_time + z, 0)

        # Calculate the optimal safety stock
        safety_stock = round(reorder_point - forecasted_demand.mean() * lead_time, 0)

        # Calculate the total cost (holding cost + stockout cost)
        holding_cost = 0.05  # it's different for every business, 0.1 is an example
        total_holding_cost = holding_cost * (initial_inventory + 0.5 * order_quantity)

        # Calculate the total cost
        total_cost = total_holding_cost

        # print("Optimal Order Quantity:", order_quantity)
        # print("Reorder Point:", reorder_point)
        # print("Safety Stock:", safety_stock)
        # print("Total Cost:", total_cost)

    # if (count != 0):
    #     print("MAPE median: ", mape_median / count, "%")
    # Print the list of product IDs with a MAPE under 100%
    # print(low_mape_product_ids)

    # Convert the dictionary to a DataFrame
    low_mape_product_ids_df = pd.DataFrame(list(order_quantities.items()), columns=['Product_ID', 'Order_Quantity'])
    low_mape_product_ids_df = low_mape_product_ids_df.sort_values(by=['Order_Quantity'], ascending=False)
    low_mape_product_ids_df = low_mape_product_ids_df.head(10)
    low_mape_product_ids_df.to_csv('order_quantities.csv', index=False)

    plt.figure(figsize=(10, 6))
    plt.bar(low_mape_product_ids_df['Product_ID'], low_mape_product_ids_df['Order_Quantity'])
    plt.xlabel('Product ID')
    plt.ylabel('Order Quantity')
    plt.title('Recommended Restock for each Product ID')
    plt.tight_layout()
    plt.savefig('toprecommendations.png')







