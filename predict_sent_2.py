
import keras
from keras.models import load_model
from keras.preprocessing import sequence 
import re
from numpy import array

from twitterscraper import query_tweets
import datetime as dt




max_review_length = 500 
NUM_WORDS=1000 # only use top 1000 words
INDEX_FROM=3   # word index offset
word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    words = sentence.split(' ')
    new_sentence = []
    for word in words:
        if word in word_to_id.keys():
            new_sentence.append(word)
    full_str = ' '.join(new_sentence)
    return full_str 

#reload model
loaded_model =load_model("C:\github\sentiment_model.h5") 

for tweet in query_tweets("#ftse", 20, begindate=dt.date(2020, 1, 18), enddate=dt.date.today())[:20]:
    text = tweet.text
    text = preprocess_text(text)
    if len(text) == 0:
        text = "neutral"
    tmp = []
    for word in text.split(" "):
        tmp.append(word_to_id[word])    
    tmp_padded = sequence.pad_sequences([tmp], maxlen=max_review_length) 
    print("%s . Sentiment: %s" % (text,loaded_model.predict(array([tmp_padded][0]))[0][0]))