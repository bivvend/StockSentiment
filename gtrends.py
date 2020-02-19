from pytrends.request import TrendReq
import datetime
from datetime import date, timedelta
import pandas as pd
import time

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

ticker_list = ['FTSE', 'AAPL','DJI','BP share price']
save_path="C:\\Github\\StockSentimentTrading\\trends_data\\trends_data.csv"

if __name__ == "__main__":
    weekday_list = n_prev_weekdays(date.today(), 800)
    # Login to Google. Only need to run this once, the rest of requests will use the same session.
    pytrend = TrendReq()
    prev_days = ['date', 'stock']
    for i in range(0, -20, -1):
        prev_days.append("day_{}".format(i))
    print(prev_days)
    interest_over_time_df = pd.DataFrame(columns = prev_days)
    for ticker in ticker_list:
        for d in weekday_list:
            date_range_string = "{} {}".format(d- timedelta(days = 25), d)
            print(date_range_string)
            failed = False
            try:
                pytrend.build_payload(kw_list = [ticker], timeframe=date_range_string)
            except Exception as e:
                print(e)
                time.sleep(10)
                try:
                    pytrend.build_payload(kw_list = [ticker], timeframe=date_range_string)
                except Exception as e2:   
                    print(e2)
                    time.sleep(60)
                    try:
                        pytrend.build_payload(kw_list = [ticker], timeframe=date_range_string)
                    except Exception as e3:
                        print(e3)
                        print("Making blank data")
                        failed = True            
            # Interest Over Time
            if failed == False:
                try:
                    response_df = pytrend.interest_over_time()
                except Exception as e:
                    print(e)
                    time.sleep(10)
                    try:
                        response_df = pytrend.interest_over_time()
                    except Exception as e2:   
                        print(e2)
                        time.sleep(60)
                        try:
                            response_df = pytrend.interest_over_time()
                        except Exception as e3:
                            print(e3)
                            print("Making blank data")
                            failed = True 
            row = []
            row.append(str(d))
            row.append(ticker)
            if failed is False:
                series = response_df[ticker].tolist()
                for i in range(0,20):
                    row.append(series[i])
                
            else:
                for i in range(0,20):
                    row.append(0)
            print(row)
            interest_over_time_df.loc[len(interest_over_time_df)] = row           

    interest_over_time_df.to_csv(save_path)