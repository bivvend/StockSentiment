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
text = "good"
text = "I had the terrible misfortune of having to view this b-movi in its entirety All I have to say is  save your time and money!!! This has got to be the worst b-movie of all time, it shouldnt even be called a b-movie, more like an f-movie! Because it fails in all aspects that make a good movie: the story is not interesting at all, all of the actors are paper-thin and not at all believable, it has bad direction and the action sequences are so fake its almost funny.......almost The movie is just packed full of crappy one-liners that no respectable person could find amusing in the least little bit This movie is supposed to be geared towards men, but all the women in it are SO utterly unattractive, especially that old wrinkled thing that comes in towards the end. They try to appear sexy in those weird, horrible costumes and they fail miserably Even some of the most ridiculous b-movies will still give you some laughs, but this is just too painful to watch!!"
text = "film"
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
    loaded_model.compile(optimizer='adam', loss='mse', metrics=['acc'])
    
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
