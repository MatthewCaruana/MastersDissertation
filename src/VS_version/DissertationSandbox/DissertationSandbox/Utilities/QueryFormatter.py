import nltk


from Utilities.CypherStringBuilder import *

class QueryFormatter:
    def __init__(self, isNeuralNet = True):
        self.Query = ""
        self.isNN = isNeuralNet

    def GenerateCypherString(self, query):
        self.Query = query

        if self.isNN:
            # Entity Detection
            tokens = nltk.word_tokenize(self.Query)
            for token in tokens:
                # Check if token is entity
                return 0
            # Entity Linking

            # Relation Prediction

            # Evidence Integration
        else:
            self.Query