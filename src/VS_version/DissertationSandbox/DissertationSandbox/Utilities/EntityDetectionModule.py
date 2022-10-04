from __future__ import unicode_literals, print_function

import sys

sys.path.insert(0, "Utilities")

import stanza
import plac
import nltk
import random
import spacy
import numpy as np
import tensorflow as tf
import tqdm


from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from nltk.chunk import conlltags2tree, tree2conlltags, ne_chunk
from nltk.tag.stanford import StanfordNERTagger
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
from nltk import pos_tag, ngrams
from spacy.training.example import Example
from spacy.util import minibatch, compounding
from tqdm import tqdm
from fuzzywuzzy import fuzz
from keras_contrib import losses

from Neo4jConnector import *


class StanzaEntityDetection:
    def __init__(self, language):
        stanza.download(language)
        self.nlp = stanza.Pipeline(lang=language, processors="tokenize,ner")
        print("Set up Stanza operator!")

    def detect(self, question):
        doc = self.nlp(question)
        print(*[f'entity: {ent.text}\ttype: {ent.type}' for ent in doc.ents], sep='\n')
        return doc.ents

class NLTKEntityDetection:
    def __init__(self):
        print("Set up NLTK operator!")

    def detect(self, question):
        ne_tree = ne_chunk(pos_tag(word_tokenize(question)))
        ne_in_sent = []
        for subtree in ne_tree:
            if type(subtree) == Tree:
                ne_label = subtree.label()
                ne_string = " ".join([token for token, pos in subtree.leaves()])
                ne_in_sent.append((ne_string, ne_label))
                
        return ne_in_sent

class SpacyEntityDetection:
    def __init__(self):
        self.model = None
        self.output_dir = "models\\EntityDetection"
        self.n_iter = 10
        print("Set up Spacy operator!")
        
    def createModel(self, training_data):
        if self.model is not None:
            self.nlp = spacy.load(model)
            print("Loaded model")
        else:
            self.nlp = spacy.blank('en')
            print("Created blank 'en' model")

        if 'ner' not in self.nlp.pipe_names:
            self.nlp.add_pipe('ner', last=True)
        else:
            self.ner = self.nlp.get_pipe('ner')

        # for _, annotations in training_data:
        #     for ent in annotations.get('entities'):
        #         self.ner.add_label(ent[2])

        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training()
            for itn in range(self.n_iter):
                section_data = training_data[:3200]
                random.shuffle(training_data)
                batches = minibatch(training_data, size=128)
                losses = {}
                for batch in tqdm(batches):
                    text, annotations = zip(*batch)
                    #doc = self.nlp.make_doc(text)
                    examples = [Example.from_dict(self.nlp.make_doc(t), a) for t,a in batch]
                    self.nlp.update(examples, drop=0.2, sgd=optimizer, losses=losses)
                print(losses)  

        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
            if not self.output_dir.exists():
                self.output_dir.mkdir()
            self.nlp.to_disk(self.output_dir)
            print("Saved model to ", self.output_dir)

class BruteForceEntityDetector:
    def __init__(self, database):
        self.database = database
        self.neo4jConnector = Neo4jConnector(uri="bolt://localhost:7687", user="admin", pwd="password")

    def detect(self, question, ngrams_range):
        # tokenize the question
        tokens = word_tokenize(question)
        ngram_list = []
        
        for i in range(ngrams_range,1,-1):
            ngram_list.extend(ngrams(question.split(), i))

        #ngram_list.sort(reverse=True)

        closest_entity = None

        # Check if entities extracted exist in database
        for entity in ngram_list:
            joined_entity = " ".join(entity)
            neo4j_entities = self.neo4jConnector.entities_with_name(joined_entity, self.database)

            if neo4j_entities != None:
                for neo4j_entity in neo4j_entities:
                    # Get similarity ratio
                    current_ratio = fuzz.ratio(neo4j_entity[0]['Text'], entity) / 100

                    if current_ratio == 100:
                        # the golden standard is found stop searching
                        closest_entity = [question, neo4j_entity, current_ratio]
                        break
                    elif closest_entity != None and closest_entity[2] < current_ratio:
                        closest_entity = [question, neo4j_entity, current_ratio]
                    elif closest_entity == None:
                        closest_entity = [question, neo4j_entity, current_ratio]

                    if closest_entity[2] == 100:
                        break

        return closest_entity

class NeuralEntityDetection:
    def __init__(self, word_count):
        self.input_size = 100
        self.hidden_size = 256
        self.layer_numbers = 2
        self.rnn_dropout = 0.3
        self.label_size = 2

        np.random.seed(456)
        self.batch_size = 500
        self.iterations = 1000

        self.dense_embedding = 16 # Dimension of the dense embedding
        self.lstm_units = 16
        self.dense_units = 100
        self.word_count = word_count


    def create_model(self, mode):
        self.model = keras.Sequential()
        self.model.add(keras.Input(shape=(300,)))
        if mode == "LSTM2":
            self.model.add(layers.Embedding(self.word_count, self.dense_embedding, embeddings_initializer="uniform", input_length=300))
            self.model.add(layers.Bidirectional(layers.LSTM(self.lstm_units, recurrent_dropout=self.rnn_dropout, return_sequences=True)))

            self.model.add(layers.Activation("relu"))
            self.model.add(layers.BatchNormalization(epsilon = 1e-05, momentum=0.1))
            self.model.add(layers.Activation("relu"))
            self.model.add(layers.Dropout(self.rnn_dropout))
            self.model.add(layers.Dense(self.label_size))
        elif mode == "LSTM":
            self.model.add(layers.Embedding(self.word_count, self.dense_embedding, input_length=300))
            self.model.add(layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=True)))

            self.model.add(layers.Activation("relu"))
            self.model.add(layers.BatchNormalization(epsilon = 1e-05, momentum=0.1))
            self.model.add(layers.Activation("relu"))
            self.model.add(layers.Dropout(self.rnn_dropout))
            self.model.add(layers.Dense(2))


        self.model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=["accuracy"])
        self.model.summary()


    def load_model(self, location):
        self.model = keras.models.load_model(location +".h5")
        print("Model loaded!")
        self.model.summary()

    def save_model(self, location):
        self.model.save(location + ".h5")
        print("Model saved!")

    def train(self, train_x, train_y, test_x, test_y, mode, database, model_location):
        print("Starting Training for " + database + " database.")
        self.create_model(mode)

        tensor_train_x = tf.convert_to_tensor(train_x)
        tensor_train_y = tf.convert_to_tensor(train_y)

        tensor_test_x = tf.convert_to_tensor(test_x)
        tensor_test_y = tf.convert_to_tensor(test_y)

        with tf.device("/gpu:0"):
            self.model.fit(tensor_train_x, tensor_train_y, validation_data=(tensor_test_x,tensor_test_y), batch_size=10, epochs=self.iterations, verbose=1)

        self.save_model(model_location)

    def detect(self, input, mode, database):
        print("Starting Predictions for " + mode + " for " + database + " database.")

        tensor_input = tf.convert_to_tensor(input)

        results = self.model.predict(tensor_input, batch_size=128, verbose=1)

        return results

    def predict_single(self, input):
        tensor_input = tf.convert_to_tensor(input)

        results = self.model.predict(tensor_input, batch_size=64, verbose=1)
        
        return results