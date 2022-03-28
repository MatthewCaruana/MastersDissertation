import pandas as pd
import numpy as np
import nltk

from nltk.tokenize import word_tokenize

class DatasetUtils:
    @staticmethod
    def FormatSimpleQuestionsForEntityDetection(data):
        # the only required information is the original question, the EntityID, the EntityText and the Entities chart
        selected = data[[0,2,5,6]]

        data_reformated = selected.copy()
        return data_reformated
        
    @staticmethod
    def FormatSimpleQuestionsForRelationPrediction(data):
        selected = data[[0,3,5]]

        data_reformated = selected.copy()
        return data_reformated

    @staticmethod
    def FormatSimpleQuestionsForQuestionOnly(data):
        selected = data[5].values

        return selected.tolist()

    @staticmethod
    def FormatSimpleQuestionsForEntitiesOnly(data):
        selected = data[6].values

        return selected.tolist()

    @staticmethod
    def FormatSimpleQuestionsForRelationOnly(data):
        selected = data[3].values

        return selected.tolist()

    @staticmethod
    def FormatSimpleQuestionsForAnswerOnly(data):
        selected = data[4].values

        return selected.tolist()

    @staticmethod
    def dictionarise_sentences(train, valid= None, test=None):
        sentences = train
        if len(valid) > 0:
            sentences = sentences + valid
        if len(test) > 0:
            sentences = sentences + test

        unique_tokens = {}
        
        index = 1
        for sentence in sentences:
            if sentence[0] == "?":
                sentence = sentence[1:]
            sentence_tokens = word_tokenize(sentence)
            for sentence_token in sentence_tokens:
                if sentence_token not in unique_tokens:
                    unique_tokens[sentence_token] = index
                    index += 1

        return unique_tokens

    @staticmethod
    def encode_sentences(dictionary, sentences, max_size=300):
        encoded_sentences = []
        for sentence in sentences:
            if sentence[0] == "?":
                sentence = sentence[1:]
            sentence_tokens = word_tokenize(sentence)
            encoded_sentence = [0] * max_size
            index = 0
            for sentence_token in sentence_tokens:
                encoded_sentence[index] = dictionary[sentence_token]
                index += 1
            encoded_sentences.append(encoded_sentence)

        return encoded_sentences

    @staticmethod
    def encode_entities(entities, max_size=300):
        encoded_entities = []
        for entity in entities:
            token_entities = entity.split(" ")
            encoded_entity = [[1,0] for i in range(max_size)] 
            
            index = 0
            for token_entity in token_entities:
                if token_entity == "O":
                    encoded_entity[index][0] = 1 
                else:
                    encoded_entity[index][0] = 0
                    encoded_entity[index][1] = 1
                index += 1

            encoded_entities.append(encoded_entity)

        return encoded_entities