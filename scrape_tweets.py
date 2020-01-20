from twitterscraper import query_tweets
import datetime as dt

# All tweets matching either Trump or Clinton will be returned. You will get at
# least 10 results within the minimal possible time/number of requests
for tweet in query_tweets("Bitcoin", 10, begindate=dt.date(2020, 1, 18), enddate=dt.date.today())[:10]:
    print(tweet.text)