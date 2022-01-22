import argparse
import nltk
import stanza
import sys

sys.path.insert(0, "Utilities")

from QuestionAnsweringModule import *
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

def load_entities(entities_location):
    file = open(entities_location, 'r', encoding="utf-8")
    entities = []
    for line in file:
        data = line[:-1].split("\t")
        entities.append((data[0], {'entities': [(0, len(data[0]), 'I')]}))
    
    file.close()
    return entities


def main(args):
    if args.qa_method == "SPACY":
        questions = load_simple_questions(args.question_data)
    else:
        questions = load_simple_questions_raw(args.question_data)
        
    print("Questions Loaded")
    entities = load_entities(args.entities_data)
    print("Entities Loaded")

    if args.qa_method == "STANZA":
        qa_model = StanzaEntityDetection(args.language)        
    elif args.qa_method == "NLTK":
        qa_model = NLTKEntityDetection()
    elif args.qa_method == "SPACY":
        qa_model = SpacyEntityDetection()

    BFentityDetector = BruteForceEntityDetector(args.database)

    if args.setting == "Create":
        print("Creating model through " + args.qa_method)
        qa_model.createModel(entities)

    for question in questions:
        question_text = question[5]
        question_token_entities = question[6]
        question_results = question[2]

        entities = qa_model.detect(question_text)
        # if no entities are found we check "brute-force" in the knowledge-base
        if entities is None:
            entities = BFentityDetector.detect(question_text)

        print(entities)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment the creation of the Entity Detection Neural Network")
    parser.add_argument('--question_data', type=str, default='data\\QuestionAnswering\\processed_simplequestions_dataset\\', help="location of files that are needed to be Answered")
    parser.add_argument('--qa_method',type=str, default="STANZA", help="STANZA/NLTK/SPACY, STANZA is the required one")
    parser.add_argument('--result_dir',type=str, default="results\\", help="location of files for each result stage")
    parser.add_argument('--entities_data', type=str, default='data\\QuestionAnswering\\freebase_names\\FB2M.tsv')
    parser.add_argument('--language', type=str, default="en")
    parser.add_argument('--database', type=str, default="fb2m")
    #parser.add_argument('--setting', type=str, required=False)

    args = parser.parse_args()
    main(args)