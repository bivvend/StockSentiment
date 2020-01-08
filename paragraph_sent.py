import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

from numpy import array
from numpy import asarray
from numpy import zeros

import matplotlib.pyplot as plt

    
import pickle

data_path = "C:\Github\DataSets\IMDB\IMDBDataset.csv"
TAG_RE = re.compile(r'<[^>]+>')
text_col = 'review'
label_col = 'sentiment'
positive_val = "positive"
negative_val = "negative"

glove_file_location = "C:\Github\DataSets\Glove\glove.6B.100d.txt"
model_save_location = "C:\Github\DataSets\ParagraphSentModel\model.json"
weights_save_location =  "C:\Github\DataSets\ParagraphSentModel\weights.h5"
tokenizer_save_location = 'C:\\Github\DataSets\\ParagraphSentModel\\tokenizer.pickle'

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
    data_df = pd.read_csv(data_path)
    print("Input dataframe shape: {0}".format(data_df.shape))
    X = []
    sentences = list(data_df[text_col])
    #Process text to remove unwanted characters
    for sen in sentences:
        X.append(preprocess_text(sen))
    print(X[3])
    #process labels
    y = data_df[label_col]
    y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))
    #train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    #Prepare embedding layer
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1
    maxlen = 100
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    
    #use GloVe embeddings to create feature matrix
    embeddings_dictionary = dict()
    glove_file = open(glove_file_location, encoding="utf8")

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary [word] = vector_dimensions
    glove_file.close()

    #create embeddings matrix
    embedding_matrix = zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    #create RNN  (LSTM network)
    model = Sequential()
    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
    model.add(embedding_layer)
    model.add(LSTM(128))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    print(model.summary())

    history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)
    score = model.evaluate(X_test, y_test, verbose=1)
    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_save_location, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weights_save_location)
    print("Saved model to disk")

    #save tokensiser

    # saving
    with open(tokenizer_save_location, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved tokenizer to disk")

    