import argparse
import nltk
import stanza
import sys
import pickle

from fuzzywuzzy import fuzz

from nltk.tokenize import word_tokenize
from nltk import ngrams

sys.path.insert(0, "Utilities")

from EntityDetectionModule import *
from RelationPredictionModule import *
from Neo4jConnector import *
from MAPAWrapper import *

from argparse import ArgumentParser

def load_simple_questions(questions_location):
    file = open(questions_location + "valid.txt", 'r', encoding="utf-8")
    records = []
    for file_line in file:
        records.append(file_line[:-1].split("\t"))
    file.close()

    entities = []
    for question_all in records:
        question = question_all[5]
        entity = question_all[2]
        entity_start_index = question.find(entity)
        entity_end_index = len(entity) + entity_start_index
        entities.append((question, {'entities': [(entity_start_index, entity_end_index, 'Entity')]}))

    return entities

def load_simple_questions_raw(questions_location):
    file = open(questions_location + "valid.txt", 'r', encoding="utf-8")
    records = []
    for file_line in file:
        records.append(file_line[:-1].split("\t"))
    file.close()
    return records

def load_entities(entities_location):
    file = open(entities_location, 'r', encoding="utf-8")
    entities = []
    for line in file:
        data = line[:-1].split("\t")
        entities.append((data[0], {'entities': [(0, len(data[0]), 'I')]}))
    
    file.close()
    return entities

def get_ngrams(question):
    # tokenize the question
    tokens = word_tokenize(question)
    ngram_list = []
    
    for i in range(ngrams):
        ngram_list.extend(ngrams(question.split(), i))

    ngram_list.sort(key=lambda t: len(ngram_list[t]), reverse=True)

    return ngram_list


def main(args):
    print("Connecting with Neo4j Server")
    neo4jGateway = Neo4jConnector(args.neo4j_url, args.neo4j_username, args.neo4j_password)

    print("Loading questions")
    if args.qa_method == "SPACY":
        questions = load_simple_questions(args.question_data)
    else:
        questions = load_simple_questions_raw(args.question_data)
        
    print("Questions Loaded")
    entities = load_entities(args.entities_data)
    print("Entities Loaded")

    # Load Entity Detection Method
    print("Loading Entity Detector")
    if args.qa_method == "STANZA":
        qa_model = StanzaEntityDetection(args.language)        
    elif args.qa_method == "NLTK":
        qa_model = NLTKEntityDetection()
    elif args.qa_method == "SPACY":
        qa_model = SpacyEntityDetection()

    print("Loading Brute Force Entity Detector")
    # Load Brute Force Method
    BFentityDetector = BruteForceEntityDetector(args.database)

    print("Loading Relation Predictors")
    # Load Relation Prediction Methods
    if args.relation_prediction_type == "LSTM":
        rp_module = LSTMRelationPrediction()
    elif args.relation_prediction_type == "GRU":
        rp_module = GRURelationPrediction()
    elif args.relation_prediction_type == "LR":
        rp_module = LRRelationPrediction()

    print("Loading Brute Force Relation Predictor")
    # Load Brute Force Prediction Methods
    bf_module = BruteForceRelationPrediction()

    print("Loading MAPA annotator")
    mapa_wrapper = MAPAWrapper()

    if args.setting == "Create":
        print("Creating model through " + args.qa_method)
        qa_model.createModel(entities)

    print("Starting Entity Detection and Entity Linking")
    for question in questions:
        question_text = question[5]
        question_token_entities = question[6]
        question_results = question[2]

        entities = BFentityDetector.detect(question_text, args.ngrams)
        # entities = qa_model.detect(question_text)
        # # if no entities are found we check "brute-force" in the knowledge-base
        # if entities is None:
        #     entities = BFentityDetector.detect(question_text)

        print(entities)

        closest_entity = None

        # Check if entities extracted exist in database
        for entity in entities:
            neo4j_entities = neo4jGateway.entities_with_name(" ".join(entity), args.neo4j_database)

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

        # Question Entity has been linked with database
        # **************************************************************************************************
        # Relation Prediction
        print("Entity for question: " + question[2] + " has been identified")
        print("Starting Relation Prediction")

        # Load all relations incoming and outgoing from node
        print("Loading all relations for node")
        outgoing_relations = neo4jGateway.outgoing_relations_for_node(closest_entity[1], args.neo4j_database)
        incoming_relations = neo4jGateway.incoming_relations_for_node(closest_entity[1], args.neo4j_database)

        relations = rp_module.detect(question)

        if relations == None:
            relations = bf_module.detect(question)

        if args.skip_dates == False:
            # identify any additional annotations
            additional_annotations = bf_module.get_dates(question)

        


        # Relation Prediction has been completed
        # **************************************************************************************************
        # Evidence Integration
        # At this point I will have the question, the entity identified and the relation identified
        
        # First step is to check whether or not we managed to guess the entity and relation

        # If either one of them is not found, skip to next question

        # If both are found we gather the score for the entity linking and relation prediction

        # Get the object node that is linked with the highest scoring entity and relation




        # **************************************************************************************************


                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment the creation of the Entity Detection Neural Network")
    parser.add_argument('--question_data', type=str, default='data\\QuestionAnswering\\processed_simplequestions_dataset\\', help="location of files that are needed to be Answered")
    parser.add_argument('--qa_method',type=str, default="STANZA", help="STANZA/NLTK/SPACY, STANZA is the required one")
    parser.add_argument('--result_dir',type=str, default="results\\", help="location of files for each result stage")
    parser.add_argument('--entities_data', type=str, default='data\\QuestionAnswering\\freebase_names\\FB2M.tsv')
    parser.add_argument('--language', type=str, default="en")
    parser.add_argument('--database', type=str, default="fb2m")
    parser.add_argument('--ngrams', type=int, default=3)
    parser.add_argument('--setting', type=str, default="NoCreate", help="Create model by setting to 'Create', skip creation by typing anything else")

    parser.add_argument('--relation_prediction_type', type=str, default="LR", help="LSTM/GRU/LR")
    parser.add_argument('--skip_dates', type=bool, default=True)

    parser.add_argument('--neo4j_url', type=str, default="bolt://localhost:7687")
    parser.add_argument('--neo4j_username', type=str, default="admin")
    parser.add_argument('--neo4j_password', type=str, default="password")
    parser.add_argument('--neo4j_database', type=str, default="fb2m")

    args = parser.parse_args()
    print(args)
    main(args)