import pandas as pd
import numpy as np
import re

from keras.models import Sequential
from keras.models import model_from_json

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from numpy import array
from numpy import asarray
from numpy import zeros

import matplotlib.pyplot as plt
import pickle

tokenizer_save_location = 'C:\\Github\DataSets\\ParagraphSentModel\\tokenizer.pickle'
model_save_location = "C:\Github\DataSets\ParagraphSentModel\model.json"
weights_save_location =  "C:\Github\DataSets\ParagraphSentModel\weights.h5"

maxlen = 100
TAG_RE = re.compile(r'<[^>]+>')
text = "This stock is going to fail it is rubbish"

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
    return sentence

if __name__ == "__main__":
    #load json and create model
    json_file = open(model_save_location, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_save_location)
    print("Loaded model from disk")
    #compile model
    loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    
    with open(tokenizer_save_location, 'rb') as handle:
        tokenizer = pickle.load(handle)

    text = preprocess_text(text)
    instance = tokenizer.texts_to_sequences(text)

    flat_list = []
    for sublist in instance:
        for item in sublist:
            flat_list.append(item)

    flat_list = [flat_list]

    instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)

    output = loaded_model.predict(instance)
    print(output)
