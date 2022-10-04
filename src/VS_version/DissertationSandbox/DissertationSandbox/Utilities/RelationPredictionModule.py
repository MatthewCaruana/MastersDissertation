from __future__ import unicode_literals, print_function

import sys

sys.path.insert(0, "Utilities")

import stanza
import plac
import nltk
import random
import spacy
import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from nltk.chunk import conlltags2tree, tree2conlltags, ne_chunk
from nltk.tag.stanford import StanfordNERTagger
from nltk.tokenize import word_tokenize
from nltk import ngrams
from spacy.training.example import Example
from spacy.util import minibatch, compounding
from tqdm import tqdm
from tensorboard.plugins.hparams import api as hp

from Neo4jConnector import *
from MAPAWrapper import *

class RNNRelationPrediction:
    def __init__(self, lbl_size, word_count):
        print("Instantiated RNN Relation Predictor")
        self.hidden_size = 256
        self.rnn_dropout = 0.3
        self.label_size = lbl_size

        np.random.seed(456)
        self.iterations = 1000

        self.dense_embedding = 16 #64 # Dimension of the dense embedding
        self.lstm_units = 16 #64
        self.dense_units = 100
        self.word_count = word_count
        self.batch_size= 64

    def create_model(self, mode):
        print("Creating Relation Prediciton RNN Model")
        self.model = keras.Sequential()
        self.model.add(keras.Input(shape=(300,)))
        if mode == "LSTM2":
            #self.model.add(layers.Embedding(self.word_count, self.dense_embedding, embeddings_initializer="uniform"))
            self.model.add(layers.Bidirectional(layers.LSTM(self.lstm_units, recurrent_dropout=self.rnn_dropout, return_sequences=False)))

            self.model.add(layers.Activation("relu"))
            self.model.add(layers.BatchNormalization())
            #self.model.add(layers.BatchNormalization(epsilon = 1e-05, momentum=0.1))
            self.model.add(layers.Activation("relu"))
            self.model.add(layers.Dropout(self.rnn_dropout))
            #self.model.add(layers.GlobalMaxPooling1D())
            #self.model.add(layers.Dense(32))
            self.model.add(layers.Dense(self.label_size))
            self.model.add(layers.Activation("relu"))
        if mode == "LSTM":
            self.model.add(layers.Embedding(self.word_count, self.dense_embedding))
            self.model.add(layers.Bidirectional(layers.LSTM(self.dense_embedding)))
            self.model.add(layers.Activation("relu"))
            self.model.add(layers.BatchNormalization())
            #self.model.add(layers.Dense(128, activation="relu"))
            self.model.add(layers.Dropout(self.rnn_dropout))
            self.model.add(layers.Dense(self.label_size, activation="softmax"))
        if mode == "LSTM-DOI":
            self.model.add(layers.Embedding(self.word_count, self.dense_embedding))
            self.model.add(layers.Bidirectional(layers.LSTM(self.dense_embedding)))
            #self.model.add(layers.Dense(self.label_size, activation="relu"))
            self.model.add(layers.Activation("relu"))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Dense(self.label_size/2, activation="relu"))
            self.model.add(layers.Dropout(self.rnn_dropout))
            self.model.add(layers.Dense(self.label_size))
            self.model.add(layers.Activation("softmax"))

        self.model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=["accuracy"])
        self.model.summary()

    def load_model(self, location):
        self.model = keras.models.load_model(location)
        self.model.summary()

    def save_model(self, location):
        self.model.save(location + ".h5")
        print("Model saved!")

    def train(self, train_x, train_y, test_x, test_y, mode, database, model_location):
        print("Starting Relation Prediciton Training for " + database + " database.")
        self.create_model(mode)

        tensor_train_x = tf.convert_to_tensor(train_x)
        tensor_train_y = tf.convert_to_tensor(train_y)

        tensor_test_x = tf.convert_to_tensor(test_x)
        tensor_test_y = tf.convert_to_tensor(test_y)

        with tf.device("/gpu:0"):
            self.model.fit(tensor_train_x, tensor_train_y, validation_data=(tensor_test_x, tensor_test_y), batch_size=10, epochs=self.iterations, verbose=1)

        self.save_model(model_location)

    def detect(self, input, mode, database):
        print("Starting Predictions for " + mode + " for " + database + " database.")

        tensor_input = tf.convert_to_tensor(input)

        results = self.model.predict(tensor_input, batch_size=128, verbose=1)

        return results

    def predict_single(self, input):
        tensor_input = tf.convert_to_tensor(input)

        results = self.model.predict(tensor_input, batch_size=1, verbose=1)
        
        return results



class BruteForceRelationPrediction:
    def __init__(self):
        print("Instantiated Brute Force Relation Predictor")
        self.determiners = ["WRB", "WP", "WDT"]
        self.mapa_wrapper = MAPAWrapper()

    def detect(self, question):
        tokens = word_tokenize(question)
        tagged_tokens = nltk.pos_tag(tokens)

        determiner = []

        parser = nltk.ChartParser()

        for tagged_token in tagged_tokens:
            # find determiners
            if tagged_token[1] in self.determiners:
                determiner.append(tagged_token)

    def get_dates(self, question):
        # find dates
        annotations = self.mapa_wrapper.process(question)
        dates = MAPAWrapper.get_dates(annotations)

        return dates


if __name__ == "__main__":
    bf = BruteForceRelationPrediction()
    bf.detect("Where did roger marquis die in 2015?")

