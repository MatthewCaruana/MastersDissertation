from __future__ import unicode_literals, print_function

import sys

sys.path.insert(0, "Utilities")

import stanza
import plac
import nltk
import random
import spacy
import tensorflow as tf

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

from Neo4jConnector import *
from MAPAWrapper import *


class GRURelationPredictor:
    def __init__(self):
        print("Instantiated GRU Relation Predictor")

    #def detect(self, question):


class LSTMRelationPrediction:
    def __init__(self):
        print("Instantiated LSTM Relation Predictor")

    #def detect(self, question):


class LRRelationPrediction:
    def __init__(self):
        print("Instantiated LR Relation Predictor")

    #def detect(self, question):

class RNNRelationPrediction:
    def __init__(self):
        print("Instantiated RNN Relation Predictor")
        self.iterations = 100

    def create_model(self):
        print("Creating Relation Prediciton RNN Model")
        self.model = keras.Sequential()
        #self.model.add(keras.Input(shape=(300,1)))
        if mode == "LSTM":
            self.model.add(layers.Bidirectional(layers.LSTM(128), input_shape=(100,1)))
            #self.model.add(layers.Bidirectional(layers.LSTM(64)))
            #self.model.add(layers.Reshape((300,1)))

            #self.model.add(layers.BatchNormalization())
            #self.model.add(layers.Dropout(self.rnn_dropout))
            ##self.model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=(300,100)))
            self.model.add(layers.Dense(self.input_size))

        self.model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])
        self.model.summary()

    def load_model(self, location):
        self.model = keras.models.load_model(location)

    def save_model(self, location, file_name):
        self.model.save(location + file_name)

    def train(self, train_x, train_y, test_x, test_y, mode, database, model_location):
        print("Starting Relation Prediciton Training for " + database + " database.")
        self.create_model(mode)

        tensor_train_x = tf.convert_to_tensor(train_x)
        tensor_train_y = tf.convert_to_tensor(train_y)

        tensor_test_x = tf.convert_to_tensor(test_x)
        tensor_test_y = tf.convert_to_tensor(test_y)

        with tf.device("/gpu:0"):
            self.model.fit(tensor_train_x, tensor_train_y, validation_data=(tensor_test_x,tensor_test_y), batch_size=1024, epochs=self.iterations, verbose=1)

        self.save_model(model_location, "simple_model2")


class BERTRelationPrediction:
    def __init__(self):
        print("Instantiated BERT Relation Prediction")

    def detect(self, question):
        print()


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

