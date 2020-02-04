import yfinance as yf
import matplotlib.pyplot as plt
import datetime
from datetime import date, timedelta
import os

def n_prev_weekdays(adate, number_of_days_back):
    i = 1
    weekdays = []
    while i < number_of_days_back: 
        adate -= timedelta(days= 1)
        while adate.weekday() > 4: # Mon-Fri are 0-4
            adate -= timedelta(days=1)
        weekdays.append(adate)
        i += 1
    return weekdays

ticker_list = ['^FTSE', 'AAPL','^DJI','BP.L']
save_path="C:\Github\StockSentimentTrading\stock_data"

if __name__ == "__main__":
    weekday_list = n_prev_weekdays(date.today(), 800)
    for ticker in ticker_list:     
        data = yf.download(ticker,str(weekday_list[-1]),str(date.today()))
        data['diff'] = data['Close'] - data['Open']
        data['Adj Close'].plot()
        save_file = os.path.join(save_path, ticker.replace('^','') + ".csv")
        data.to_csv(save_file)