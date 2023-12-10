import numpy as np
from prophet import Prophet
from scipy import stats
import multiprocessing as mp
from scipy.stats import norm
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


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
    product_details = {}

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



        print("MAPE: ", mape, "%")
        if mape > 0:
            mape_median += mape
            count += 1

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



        # Calculate the reorder point
        reorder_point = round(forecasted_demand.mean() * lead_time + z, 0)

        # Calculate the optimal safety stock
        safety_stock = round(reorder_point - forecasted_demand.mean() * lead_time, 0)

        # Calculate the total cost (holding cost + stockout cost)
        holding_cost = 0.05  # it's different for every business, 0.1 is an example
        total_holding_cost = holding_cost * (initial_inventory + 0.5 * order_quantity)

        # Calculate the total cost
        total_cost = total_holding_cost

        if mape > 0 and mape < 100:
            # print(product_id)
            # print("MAPE: ", mape, "%")
            # Add the product ID and its order quantity to the dictionary
            product_details[product_id] = {
                'Order_Quantity': order_quantity,
                'Reorder_Point': reorder_point,
                'Safety_Stock': safety_stock,
                'Total_Cost': total_cost
            }

        # print("Optimal Order Quantity:", order_quantity)
        # print("Reorder Point:", reorder_point)
        # print("Safety Stock:", safety_stock)
        # print("Total Cost:", total_cost)


    if (count != 0):
        print("MAPE median: ", mape_median / count, "%")
    # Print the list of product IDs with a MAPE under 100%


    # Convert the dictionary to a DataFrame
    product_details_df = pd.DataFrame.from_dict(product_details, orient='index')
    product_details_df = product_details_df.sort_values(by=['Order_Quantity'], ascending=False)
    product_details_df = product_details_df.head(10)
    # print(product_details_df)

# Try to import OpenAI's GPT-3 for generating recommendations
import openai
openai.api_key = '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'

    # Function to generate a resupply recommendation message using OpenAI's API
def generate_resupply_message(product_id, order_quantity, reorder_point, safety_stock):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Generate a resupply recommendation message for the Stock Manager of Product ID {product_id}, advising them to order {order_quantity} units of the product, as the stock has reached the reorder point of {reorder_point} units. Additionally, suggest purchasing an extra {safety_stock} units as safety stock to prevent any gaps in supply."}
        ]
    )
    # Extract the message from the response
    message = response.choices[0].message['content']
    return message

for index, row in product_details_df.iterrows():
    product_id = index
    order_quantity = row['Order_Quantity']
    reorder_point = row['Reorder_Point']
    safety_stock = row['Safety_Stock']
    total_cost = row['Total_Cost']
    # Now you can use these variables in your code
    # print(f"Product ID: {product_id}, Order Quantity: {order_quantity}, Reorder Point: {reorder_point}, Safety Stock: {safety_stock}, Total Cost: {total_cost}")
    resupply_message = generate_resupply_message(product_id, order_quantity, reorder_point, safety_stock)
    print(ResourceWarning)
    with open('restock_info.txt','w') as output_file:
        output_file.write(resupply_message)








