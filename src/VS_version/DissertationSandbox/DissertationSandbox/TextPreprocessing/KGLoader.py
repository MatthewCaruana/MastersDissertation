import sys

sys.path.insert(0, "Utilities")

import neo4j
import os
import glob
import json

from openie import StanfordOpenIE
from py2neo import Graph, Node, Relationship

from Neo4jConnector import *
from Triples import *
from Property import *
from Py2NeoConnector import *

def populateMaltaBudgetsData(connector, databaseName="maltabudgets"):
    location = "data/MaltaBudgets/English/triples/"
    files = [f for f in os.listdir(location)]

    for file in files:
        json_file = open(location + file, 'r')
        json_content = json.loads(json_file.read())

        year = file[1:5]

        properties = [Property("Year", year)]

        for triple in json_content:
            if triple:
                triple_neo4j = Triple(str(triple[0]["subject"]), triple[0]["relation"], triple[0]["object"])
                connector.AddTriple(triple_neo4j, properties)

def populateFreebaseData(connector, databaseName="freebase"):
    for file in glob.glob("data/QuestionAnswering/SimpleQuestions_v2/freebase-subsets/FB2M.json"):
        json_file = open(file, 'r')
        json_content = json.loads(json_file.read())
        properties = []
        entities = set()
        relations = set()

        i = 0
        for triple in json_content:
            if i % 10000 == 0:
                print("Went through " + str(i) + " / " + str(len(json_content)))

            entities.add(str(triple["subject"]))
            entities.add(str(triple["object"]))
            relations.add(str(triple["relation"]))
            i = i +1

        entities_list = list(entities)
        print("Entities " + str(len(entities_list)))

        #connector.AddNodes(entities_list, ["name"])

        for relation in relations:
            relations_list = []
            for triple in json_content:
                if str(triple["relation"]) == relation:
                    relations_list.append((str(triple["subject"]),{}, str(triple["object"])))
            
            print ("Relation: " + relation)
            print("Total: " + str(len(relations_list)))

            if not connector.RelationExists(relation, len(relations_list)):
                connector.AddRelationsAndNodesWithCypher(relations_list, relation)
            else:
                print(relation + " already exists in Graph")

        # i = 0
        # for triple in json_content:
        #     if i % 10000 == 0:
        #         print("Went through " + str(i) + " / " + str(len(json_content)))
        #     connector.AddRelationWithNodes(str(triple["subject"]),str(triple["relation"]), str(triple["object"]))
        #     i = i + 1


def main():
    malta_graph = Py2NeoConnector("bolt://localhost:7687", username="admin", password="password", database="maltabudgets")
    #freebase_graph = Py2NeoConnector("bolt://localhost:7687", username="admin", password="password", database="fb5m")
    
    populateMaltaBudgetsData(malta_graph)
    #populateFreebaseData(freebase_graph)


if __name__ == "__main__":
    main()