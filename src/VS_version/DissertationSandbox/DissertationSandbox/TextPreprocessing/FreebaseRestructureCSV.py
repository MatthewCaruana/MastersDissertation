import sys

sys.path.insert(0, "Utilities")

from Triples import *
import argparse
import pickle
import json
import csv

def main(args):
    print("Opening pickle file for " + args.type)
    graph_file = open(args.graph_data, 'rb') 

    print("Opening Entities file for " + args.type)
    entity_file = open(args.entity_location, 'r', encoding="utf-8")

    entity = {}
    csv_entities = {}

    i = 0
    for line in entity_file:
        if i % 1000000 == 0:
            print("line: {}".format(i))

        link = line.split("\t")[0]
        fb_type = line.split("\t")[1]
        text = line.split("\t")[2]

        entity[link] =  { "type": fb_type, "text": text}
        i = i+1


    print("Unpickling file")
    freebase_triples = pickle.load(graph_file)
    updated_triples = []

    i = 0
    for _subject, _relation in freebase_triples:
        if i % 1000 == 0:
            print("line: {}".format(i))

        _object = list(freebase_triples[(_subject,_relation)])[0]

        #Convert freebase notation (with links) to raw text
        subject_text = ""
        subject_type = ""
        relation_text = ""
        object_text = ""
        object_type = ""

        #Convert Subject
        if _subject in entity:
            if len(entity[_subject]) > 2:
                print("more than one entry for subject " + _subject)
                subject_text = entity[_subject][0]['text'][:-1]
                subject_type = entity[_subject][0]['type']
            else:
                subject_text = entity[_subject]['text'][:-1]
                subject_type = entity[_subject]['type']

        # Check if subject is already used as part of structure
        subject_id = 0
        if subject_text not in csv_entities:
            subject_id = len(csv_entities) + 1
            csv_entities[subject_text] = {"id": len(csv_entities) + 1, "type": subject_type, "text": subject_text }
        else:
            subject_id = csv_entities[subject_text]["id"]
        
        #Restructure Relation
        relation_text = _relation.split(".")[-1]

        #Convert Object
        if _object in entity:
            if len(entity[_object]) > 2:
                print("more than one entry for object" + _object)
                object_text = entity[_object][0]['text'][:-1]
                object_type = entity[_object][0]['type']
            else:
                object_text = entity[_object]['text'][:-1]
                object_type = entity[_object]['type']

        object_id = 0
        #if next((item for item in csv_entities if csv_entities[item]["text"] == object_text), None) == None:
        if object_text not in csv_entities:
            object_id = len(csv_entities) + 1
            csv_entities[object_text] = {"id": len(csv_entities) + 1, "type" : object_type, "text" : object_text }
        else:
            object_id = csv_entities[object_text]["id"]

        #Store result as Triple
        triple = {"subject": subject_id, "relation": relation_text, "object": object_id}
        updated_triples.append(triple)
        i = i + 1

    sorted_entities = sorted(csv_entities.items())

    if len(sorted_entities) > 100000:
        entities_chunks = [sorted_entities[x:x+100000] for x in range(0,len(sorted_entities), 100000)]
    else:
        entities_chunks = [csv_entities]

    if len(updated_triples) > 100000:
        relation_chunks = [updated_triples[x:x+100000] for x in range(0, len(updated_triples), 100000)]
    else:
        relation_chunks = [updated_triples]

    i = 0
    for chunk in entities_chunks:
        print("Creating " + args.output + args.type + "_entities" + str(i) + ".csv")
        f_entities_output = open(args.output + args.type + "_entities" + str(i) + ".csv", 'w', newline='', encoding="utf-8")
        csv_writer = csv.writer(f_entities_output)
        header_entities = ["id", "text", "type"]
        csv_writer.writerow(header_entities)
        for entity in chunk:
            csv_writer.writerow([entity[1]["id"], entity[1]["text"], entity[1]["type"]])
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
    parser.add_argument('--triples_location', type=str, default='data\\QuestionAnswering\\SimpleQuestions_v2\\freebase-subsets\\freebase-FB2M.txt')
    parser.add_argument('--entity_location', type=str, default='data\\QuestionAnswering\\freebase_names\\names.trimmed.2M.txt')
    parser.add_argument('--graph_data',type=str, default="data\\QuestionAnswering\\SimpleQuestions_v2\\freebase-subsets\\freebase-FB2M-graph.pk")
    parser.add_argument('--output', type=str, default="data\\QuestionAnswering\\SimpleQuestions_v2\\freebase-subsets\\")
    parser.add_argument('--type', type=str, default="FB2M")

    args = parser.parse_args()

    main(args)