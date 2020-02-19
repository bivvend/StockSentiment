
import pandas as pd
import datetime
import os

stock_path = "c:\github\stocksentimenttrading\stock_data"
sentiment_df = "c:\github\stocksentimenttrading\\trends_data\\trends_data.csv"
output_df = "c:\github\stocksentimenttrading\merged_data\merged_data.csv"
ticker_list = ['FTSE', 'AAPL','DJI','BP.L']
num_days_back = 20

#load sentiment df
sent_df = pd.read_csv(sentiment_df)
print(sent_df.head())
#Merge all data into one DataFrame
#load all stock
col_list = []
col_list.append("Date")
col_list.append("Stock_name")
for i in range(0, num_days_back):
    val = i* -1
    col_list.append("change_day_{0}".format(val))
    col_list.append("volume_day_{0}".format(val))
    col_list.append("sentiment_day_{0}".format(val))
full_df = pd.DataFrame(columns=col_list)
print(col_list)
for ticker in ticker_list:
    path = os.path.join(stock_path, "{0}.csv".format(ticker))
    temp_df = pd.read_csv(path)
    change_range = temp_df["diff"].max() - temp_df["diff"].min()
    volume_range = temp_df["Volume"].max() - temp_df["Volume"].min()
    print((change_range, volume_range))    
    print(len(temp_df))

    for i in range(num_days_back, len(temp_df), 1):
        row_data = [] 
        #find correct row in sentiment df
        date_i = str(temp_df['Date'].iloc[i])
        sent_row = sent_df.loc[(sent_df['date'] == date_i) & (sent_df['stock'].str.contains(ticker))]
        row_data.append(temp_df['Date'].iloc[i])
        row_data.append(ticker)
        for d in range(0, num_days_back):
            if  i - d >= 0 and i - d < len(temp_df) :
                val = d* -1
                sent_col_name = "day_{0}".format(val)
                change_val = temp_df['diff'].iloc[i -d]/change_range
                vol_val = temp_df['Volume'].iloc[i - d]/volume_range
                row_data.append(change_val)
                row_data.append(vol_val)                
                try:
                    sent_vals = sent_row[sent_col_name].values
                    sent_val = sent_vals[0] / 100.0
                except Exception as e:
                    sent_val = 0
                row_data.append(sent_val)
        full_df.loc[(len(full_df))] = row_data
print(full_df.head())


full_df.to_csv(output_df)





    




