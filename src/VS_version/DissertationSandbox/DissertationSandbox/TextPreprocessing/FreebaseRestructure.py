import sys

sys.path.insert(0, "Utilities")

from Triples import *
import argparse
import pickle
import json


def main(args):
    print("Opening pickle file for " + args.type)
    graph_file = open(args.graph_data, 'rb') 

    print("Opening Entities file for " + args.type)
    entity_file = open(args.entity_location, 'r', encoding="utf-8")

    entity = {}

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
        relation_text = ""
        object_text = ""
        #Convert Subject
        if _subject in entity:
            if len(entity[_subject]) > 2:
                print("more than one entry for subject " + _subject)
                subject_text = entity[_subject][0]['text'][:-1]
            else:
                subject_text = entity[_subject]['text'][:-1]
        
        #Restructure Relation
        relation_text = _relation.split(".")[-1]

        #Convert Object
        if _object in entity:
            if len(entity[_object]) > 2:
                print("more than one entry for object" + _object)
                object_text = entity[_object][0]['text'][:-1]
            else:
                object_text = entity[_object]['text'][:-1]

        #Store result as Triple
        triple = Triple(subject_text, relation_text, object_text)
        updated_triples.append(triple)
        i = i + 1

    foutput = open(args.output + args.type + ".json", 'w')
    foutput.write(json.dumps([triple.__dict__ for triple in updated_triples]))
    foutput.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transforming the Freebase data to our Triple structure")
    parser.add_argument('--triples_location', type=str, default='data\\QuestionAnswering\\SimpleQuestions_v2\\freebase-subsets\\freebase-FB5M.txt')
    parser.add_argument('--entity_location', type=str, default='data\\QuestionAnswering\\freebase_names\\names.trimmed.5M.txt')
    parser.add_argument('--graph_data',type=str, default="data\\QuestionAnswering\\SimpleQuestions_v2\\freebase-subsets\\freebase-FB5M-graph.pk")
    parser.add_argument('--output', type=str, default="data\\QuestionAnswering\\SimpleQuestions_v2\\freebase-subsets\\")
    parser.add_argument('--type', type=str, default="FB5M")

    args = parser.parse_args()

    main(args)