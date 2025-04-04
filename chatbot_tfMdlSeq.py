import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder # type: ignore
import tensorflow as tf
from tensorflow import keras # type: ignore

#------------ Predefined functions -----------------------------#

def bow(vocabulary:list, sentence:str, )-> list:
    '''Generates a bow-array:list[int] for a given sentence:str based on a given vocabulary:list[str] '''
    bagOfWords = []
    for word in vocabulary:
        if word in sentence.lower():
            bagOfWords.append(1)
        else:
            bagOfWords.append(0)
    return bagOfWords

def bow_list(vocabulary:list, sentences:list[str])->list[int]:
    ''' Generates a list of bow-arrays for list of sentences based on a given vocabulary:list[str] '''
    bow_list:list[int] = []
    for sentence in sentences:
        bow_list.append(bow(vocabulary, sentence))
    return bow_list

def remove_chars(chars, sentences):
    ''' Removes chars from sentences:list[str] and return "cleaned" sentences '''
    clean_sentences = []
    for sentence in sentences:
        for char in chars:
            if char in sentence:
                sentence = sentence.replace(char, '')
        sentence = sentence.strip().lower()
        clean_sentences.append(sentence)
    return clean_sentences

def create_vocabulary(sentences):
    ''' Creates array of words (vocabulary) based on given sentences '''
    words_list = []
    for sentence in sentences:
        words = sentence.split(" ")
        for word in words:
            words_list.append(word)
    vocabulary = list(set(words_list))
    # vocabulary = words_list
    return vocabulary


# ------------------- load data ----------------------------------------------#
# Load the dataset
df = pd.read_csv('ChatbotTraining.csv',
                sep=',',              # Specify separator (default is comma)
                encoding='utf-8',     # Specify encoding
                header=0)
# print(df['tag'].shape)
# print(df['patterns'].shape)


#------------------- input data ----------------------------------------------#
sentences = df['patterns'].values.tolist()
chars = '!?.\','
sentences = remove_chars(chars, sentences)

#------------------- vocabulary ----------------------------------------------#
vocabulary = create_vocabulary(sentences)
X_train = bow_list(vocabulary, sentences)
X_train = np.array(X_train) # Ensure X_train is a NumPy array

# ------------------- labels / output data ----------------------------------#
labels = df['tag'].values.tolist()
print("Possible labels/categories:", np.unique(labels))
le = LabelEncoder()
y = le.fit_transform(labels)

#------------------ Define TF-Model ------------------------------------------#
tf_chatbot = tf.keras.Sequential()
tf_chatbot.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],))) # input layer
tf_chatbot.add(tf.keras.layers.Dense(128, activation='relu')) # hidden layer
tf_chatbot.add(tf.keras.layers.Dense(len(set(y)), activation='softmax')) # output layer

#---------------- Compile model ---------------------------------------------#
tf_chatbot.compile(optimizer='adam',                        # popular optimizer for training
                   loss='sparse_categorical_crossentropy',  # loss function for multi-class classification
                   metrics=['accuracy'])                    # accuracy metric

#------------------ Train model ---------------------------------------------#
tf_chatbot.fit(X_train, y, batch_size=8, epochs=10) # fit model to data)

#------------------ Testing --------------------------------------------------#
# Using train data for testing
X_test = X_train
_, accuracy = tf_chatbot.evaluate(X_test, y)
print(f"Accuracy: {accuracy * 100:.2f}%")

