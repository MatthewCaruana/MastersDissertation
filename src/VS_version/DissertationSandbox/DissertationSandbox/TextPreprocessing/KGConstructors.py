# Converts array of sentences to KG structure of a triple {subject, relation, object}

import sys

sys.path.insert(0, "Utilities")

import os
import stanza
from stanza.server import CoreNLPClient
import glob
import json
import nltk
import time
import argparse
import csv
import time

from EntityExtractor import *
from Triples import *

def Reconstruct(all_triples):
    all_entities = {}
    all_relations = []

    entity_id = 0
    for (year, triples) in all_triples:
        for triple in triples:
            if triple["subject"] not in all_entities:
                subject_id = entity_id
                all_entities[triple["subject"]] = {"id":entity_id, "text":triple["subject"], "year": year}
                entity_id = entity_id + 1
            else:
                subject_id = all_entities[triple["subject"]]["id"]
            if triple["object"] not in all_entities:
                object_id = entity_id
                all_entities[triple["object"]] = {"id":entity_id, "text":triple["object"], "year":year}
                entity_id = entity_id + 1
            else:
                object_id = all_entities[triple["object"]]["id"]
            all_relations.append({"subject_id": subject_id, "relation": triple["relation"], "object_id": object_id})


    sorted_entities = sorted(all_entities.items())

    print("Creating all_entities.csv")
    f_entities_output = open(args.location + "all_entities.csv", 'w', newline='', encoding="utf-8")
    csv_writer = csv.writer(f_entities_output)
    header_entities = ["id", "text", "year"]
    csv_writer.writerow(header_entities)
    for entity in all_entities:
        csv_writer.writerow([all_entities[entity]["id"], all_entities[entity]["text"], all_entities[entity]["year"]])
    f_entities_output.close()

    print("Creating all_relations.csv")
    f_relations_output = open(args.location + "all_locations.csv", "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(f_relations_output)
    header_relations = ["subject_id", "relation", "output_id"]
    csv_writer.writerow(header_relations)
    for relation in all_relations:
        csv_writer.writerow([relation["subject_id"], relation["relation"], relation["object_id"]])
    f_relations_output.close()

    print("Finished")


def main(args):
    stanza.install_corenlp()

    start_time = time.time()

    if args.extraction == "STANZA":
        extractor = StanzaEntityExtractor(args.language)
    elif args.extraction == "NLTK":
        extractor = NLTKEntityExtractor()

    all_triples = []

    if args.skip:
        files = [f for f in os.listdir(args.result_dir)]

        for file in files:
            print("Getting info from " + args.result_dir + file)
            f = open(args.result_dir + file,'r', encoding="utf-8")

            year = file[1:5]

            file_triples = json.loads(f.read())
            all_triples.append((year, file_triples))

            f.close()
    else:
        files = [f for f in os.listdir(args.location)]

        triples_count = 0

        with CoreNLPClient(annotators=["openie"], be_quiet=True) as openie_client:
            for file in files:
                sentence_list = []
                print("Getting info from " + args.location + file)
                f = open(args.location + file,'r', encoding="utf-8")

                file_triples = []
                for line in f:
                    sentence_list.append(json.loads(line))

                year = file[1:5]

                if len(sentence_list) > 0:
                    count = 0
                    for sentence in sentence_list[0]:
                        entities = extractor.extract_entities(sentence)
    
                        triples = []
    
                        ann = openie_client.annotate(sentence)
                        if len(entities) > 0:
                            for sentence in ann.sentence:
                                for triple in sentence.openieTriple:
                                    file_triples.append(Triple(triple.subject, triple.relation, triple.object))
                        count = count + 1

                        triples_count = triples_count + len(file_triples)
    
                    json_file = open(args.result_dir + file.replace(".json", "-triples.json"), 'w')
                    json_file.write(json.dumps([triple.__dict__ for triple in file_triples]))
                    json_file.close()

                    stats_file = open(args.location + str(count) + "-statistics.txt", "w")
                    stats_file.write("Triples Generated: " + str(triples_count))
                    stats_file.write("Time Taken: " + str(time.time()-start_time) + " in seconds")
                    stats_file.close()

                    all_triples.append((1, file_triples))
    
                    f.close()

    

    # create mega files
    Reconstruct(all_triples)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment the creation of the Entity Detection Neural Network")
    parser.add_argument('--location', type=str, default='data\\WashingtonPost\\data\\sentence_blocks_unicode\\')
    parser.add_argument('--extraction',type=str, default="STANZA", help="STANZA/NLTK, STANZA is the required one")
    parser.add_argument('--result_dir',type=str, default="data\\WashingtonPost\\data\\sentence_blocks_unicode\\triples\\", help="location of files for each result stage")
    parser.add_argument('--language', type=str, default="en")
    parser.add_argument('--rel_extraction', type=str, default="OPENIE")
    parser.add_argument('--skip', type=bool, default=False)

    args = parser.parse_args()
    main(args)