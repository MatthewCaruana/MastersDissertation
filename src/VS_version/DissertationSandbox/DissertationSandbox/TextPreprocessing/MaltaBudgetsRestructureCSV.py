import sys
from os import listdir
from os.path import isfile, join

sys.path.insert(0, "Utilities")

from Triples import *
import argparse
import pickle
import json
import csv

def main(args):
    print("Getting list of all Malta Budget files")
    files = [f for f in listdir(args.triples_location) if isfile(join(args.triples_location, f))]

    all_entities = {}
    all_relation_types = {}
    all_triples = []


    for file in files:
        print("Opening " + file)
        year = file[1:5]
        open_file = open(args.triples_location + file, 'r')
        triples_data = json.load(open_file)
        for triple in triples_data:
            subject_id = 0
            if triple["subject"] not in all_entities:
                subject_id = len(all_entities) + 1
                all_entities[triple["subject"]] = {"ID": subject_id, "Text": triple["subject"], "Year": year}
            else:
                subject_id = all_entities[triple["subject"]]["ID"]

            object_id = 0
            if triple["object"] not in all_entities:
                object_id = len(all_entities) + 1
                all_entities[triple["object"]] = {"ID": object_id, "Text": triple["object"], "Year": year}
            else:
                object_id = all_entities[triple["object"]]["ID"]

            triple = {"subject": subject_id, "relation": triple["relation"], "object": object_id}
            all_triples.append(triple)

    sorted_entities = sorted(all_entities.items())

    if len(sorted_entities) > 100000:
        entities_chunks = [sorted_entities[x:x+100000] for x in range(0,len(sorted_entities), 100000)]
    else:
        entities_chunks = [all_entities]

    if len(all_triples) > 100000:
        relation_chunks = [all_triples[x:x+100000] for x in range(0, len(all_triples), 100000)]
    else:
        relation_chunks = [all_triples]

    i = 0
    for chunk in entities_chunks:
        print("Creating " + args.output + args.type + "_entities" + str(i) + ".csv")
        f_entities_output = open(args.output + args.type + "_entities" + str(i) + ".csv", 'w', newline='', encoding="utf-8")
        csv_writer = csv.writer(f_entities_output)
        header_entities = ["id", "text", "year"]
        csv_writer.writerow(header_entities)
        for entity in chunk:
            csv_writer.writerow([chunk[entity]["ID"], chunk[entity]["Text"], chunk[entity]["Year"]])
        f_entities_output.close()
        i = i + 1

    i = 0
    for chunk in relation_chunks:
        print("Creating " + args.output + args.type + "_relations" + str(i) + ".csv")
        f_relations_output = open(args.output + args.type + "_relations" + str(i) + ".csv", "w", newline='', encoding="utf-8")
        csv_writer = csv.writer(f_relations_output)
        header_relations = ["subject_id", "relation", "object_id"]
        csv_writer.writerow(header_relations)
        for relation in chunk:
            csv_writer.writerow([relation["subject"], relation["relation"], relation["object"]])
        f_relations_output.close()
        i = i+ 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transforming the Freebase data to our Triple structure")
    parser.add_argument('--triples_location', type=str, default='data\\MaltaBudgets\\English\\triples\\')
    parser.add_argument('--output', type=str, default="data\\MaltaBudgets\\English\\triples\\")
    parser.add_argument('--type', type=str, default="MaltaBudgets")

    args = parser.parse_args()

    main(args)