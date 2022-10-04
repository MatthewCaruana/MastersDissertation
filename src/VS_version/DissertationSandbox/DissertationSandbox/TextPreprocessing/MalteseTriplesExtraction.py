import argparse
import sys
import pandas as pd
import tqdm
import json
import os.path
import csv
import re
import conllu
import time

import stanza
from stanza.server import CoreNLPClient

from transformers import pipeline

sys.path.insert(0, "Utilities")

from EntityExtractor import *
from Triples import *


def handle_DOI(args):
    print("Loading data")
    doi_file = open(args.data_location, 'r', encoding="utf-8")
    doi_data = json.load(doi_file)
    doi_file.close()

    maltese_doi_sentences = []
    english_doi_sentences = []
    maltese_doi_extractions = []
    english_doi_extractions = []

    for single_doi in doi_data:
        split_sentences = re.split(r"\.|\?|\!", single_doi["Content"])
        if "en" in single_doi["Number"]:
            #if single_doi["Date"][6:] == "1997":
            english_doi_sentences.extend(split_sentences)
        else:
            #if single_doi["Date"][6:] == "2022":
            maltese_doi_sentences.extend(split_sentences)

    print("Saving Maltese Content")
    if args.skip_mt == False:
        print("Saving Maltese DOI sentences")
        if not os.path.isfile(args.result_location + "mt\\sentences-2022.txt"):
            with open(args.result_location + "mt\\sentences-2022.txt", 'w', encoding="utf-8") as file:
                json.dump(maltese_doi_sentences, file, ensure_ascii = False)

        maltese_doi_extractions = load_doi_dependency_tree(args.maltese_doi_data_location)


        print("Saving Maltese DOI triple extractions")
        if not os.path.isfile(args.result_location + "mt\\triples.json"):
            with open(args.result_location + "mt\\triples.json", 'w', encoding="utf-8") as file:
                json.dump(maltese_doi_extractions, file, ensure_ascii = False)

        print("Starting creating entities and relation files")
        entities = {}
        relations = []

        current_count = 0
        for extractions in maltese_doi_extractions:
            for triple in extractions["triples"]:
                if triple[0] not in entities:
                    subject_id = current_count
                    entities[triple[0]] = {'id':subject_id, 'text': triple[0]}
                    current_count = current_count + 1
                if triple[2] not in entities:
                    object_id = current_count
                    entities[triple[2]] = {'id':object_id, 'text': triple[2]}
                    current_count = current_count + 1

                triple_content = {"subject": subject_id, "relation": triple[1], "object": object_id, "sentence": extractions["sentence"]}
                relations.append(triple_content)

        print("Saving entities and relations separately")
        f_entities_output = open(args.result_location + "mt\\DOI_entities.csv", 'w', newline='', encoding="utf-8")
        csv_writer = csv.writer(f_entities_output)
        header_entities = ["id", "text"]
        csv_writer.writerow(header_entities)
        for entity in entities:
            csv_writer.writerow([entities[entity]["id"], entities[entity]["text"]])
        f_entities_output.close()

        f_relations_output = open(args.result_location + "mt\\DOI_relations.csv", 'w', newline='', encoding="utf-8")
        csv_writer = csv.writer(f_relations_output)
        header_entities = ["subject", "relation","object", "sentence"]
        csv_writer.writerow(header_entities)
        for relation in relations:
            csv_writer.writerow([relation["subject"], relation["relation"], relation["object"], relation["sentence"]])
        f_relations_output.close()

    print("Saving English Content")
    if args.skip_eng == False:
        start_time = time.time()
        english_doi_extractions = extract_english_triples(english_doi_sentences)

        stats_file = open(args.result_location + "statistics.txt", "w")
        stats_file.write("Triples Generated: " + str(len(english_doi_extractions)))
        stats_file.write("Time Taken: " + str(time.time()-start_time) + " in seconds")
        stats_file.close()

        print("Saving English DOI sentences")
        if not os.path.isfile(args.result_location + "en\\sentences-all.txt"):
            with open(args.result_location + "en\\sentences-all.txt", 'w', encoding="utf-8") as file:
                json.dump(english_doi_sentences, file, ensure_ascii = False)

        print("Saving English DOI triple extractions")
        if not os.path.isfile(args.result_location + "en\\triples-all.json"):
            with open(args.result_location + "en\\triples-all.json", 'w', encoding="utf-8") as file:
                extraction_string = json.dumps([extraction.__dict__ for extraction in english_doi_extractions])
                file.write(extraction_string)

        entities = {}
        relations = []

        current_count = 0
        
        print("Starting creating entities and relation files")
        for extraction in english_doi_extractions:
            if extraction.subject not in entities:
                subject_id = current_count
                entities[extraction.subject] = {"id":subject_id, "text": extraction.subject}
                current_count = current_count + 1

            if extraction.object not in entities:
                object_id = current_count
                entities[extraction.object] = {"id":object_id, "text": extraction.object}
                current_count = current_count + 1

            triple = {"subject": subject_id, "relation": extraction.relation, "object": object_id, "sentence": extraction.sentence}
            relations.append(triple)

        print("Saving entities and relations separately")
        f_entities_output = open(args.result_location + "en\\DOI_entities.csv", 'w', newline='', encoding="utf-8")
        csv_writer = csv.writer(f_entities_output)
        header_entities = ["id", "text"]
        csv_writer.writerow(header_entities)
        for entity in entities:
            csv_writer.writerow([entities[entity]["id"], entities[entity]["text"]])
        f_entities_output.close()

        f_relations_output = open(args.result_location + "en\\DOI_relations.csv", 'w', newline='', encoding="utf-8")
        csv_writer = csv.writer(f_relations_output)
        header_entities = ["subject", "relation","object", "sentence"]
        csv_writer.writerow(header_entities)
        for relation in relations:
            csv_writer.writerow([relation["subject"], relation["relation"], relation["object"], relation["sentence"]])
        f_relations_output.close()


def load_doi_dependency_tree(location):
    sentences = []
    file_names = [f for f in os.listdir(location) if os.path.isfile(os.path.join(location, f))]

    for file_name in file_names:
        file = open(location + file_name, 'r', encoding="utf-8")
        for row in conllu.parse_incr(file):
            tree = row.to_tree()
            triples = extract_maltese_triples(tree)
            if not triples == []:
                sentence = " ".join([item['form'] for item in row])
                sentences.append({"sentence": sentence, "triples": triples})
        file.close()

    return sentences

def extract_maltese_triples(dependency_tree):
    rootNode = []
    subjectNodes = []
    objectNodes = []

    triples = []

    # locate root
    rootNode = dependency_tree.token
    children = dependency_tree.children
    for child in children:
        if child.token['deprel'] == "nsubj":
            subjectNodes.append(child.token)
        elif child.token['deprel'] == "obj":
            objectNodes.append(child.token)

    for subjectNode in subjectNodes:
        for objectNode in objectNodes:
            triples.append([subjectNode["form"], rootNode["form"], objectNode["form"]])

    return triples


def extract_english_triples(sentences):
    stanza.install_corenlp()
    extractor = StanzaEntityExtractor("en")

    triples_list = []

    with CoreNLPClient(annotators=["openie"], be_quiet=True) as openie_client:
        for orig_sentence in sentences:
            sentence_triples = []
            entities = extractor.extract_entities(orig_sentence)

            stanzaAnnotation = openie_client.annotate(orig_sentence)
            if len(entities) > 0:
                for sentence in stanzaAnnotation.sentence:
                    for triple in sentence.openieTriple:
                        sentence_triples.append(Triple(triple.subject, triple.relation, triple.object, orig_sentence))
            triples_list.extend(sentence_triples)

    return triples_list

def main(args):
    if args.dataset == "DOI":
        print("Starting triples extraction of DOI data")
        handle_DOI(args)
    elif args.dataset == "Budgets":
        print("Starting triples extraction of Budgets data")
        handle_budgets(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraction of Maltese Triples from our datasources which are DOI and Budget Data")
    parser.add_argument('--data_location', type=str, default='data\\DOI\\data\\DOIContent.json')
    parser.add_argument('--maltese_doi_data_location', type=str, default='data\\DOI\\triples\\mt\\DEP\\final\\')
    parser.add_argument('--result_location', type=str, default='data\\DOI\\triples\\')
    parser.add_argument('--dataset', type=str, default="DOI")
    parser.add_argument('--skip_eng', type=bool, default=True)
    parser.add_argument('--skip_mt', type=bool, default=False)

    args = parser.parse_args()
    main(args)
