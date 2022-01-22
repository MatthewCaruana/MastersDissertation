# this is just as a test file with very standard data
import neo4j
import os
import glob
import json

from openie import StanfordOpenIE
from py2neo import Graph, Node, Relationship

import sys

sys.path.insert(0, "Utilities")

from Neo4jConnector import *
from Triples import *
from Property import *
from Py2NeoConnector import *
from CypherStringBuilder import *
from Transformations import *

kg_list =[{"subject": "Lower dividends", "relation": "are expected", "object": "to materialise"},
          {"subject": "Donald Trump", "relation": "is", "object": "president of the United States of America"},
          {"subject": "John Mallia", "relation": "worked as", "object": "a TV presenter"},
          {"subject": "Obesity", "relation": "is killing", "object": "100 million every year"}]

def populateMaltaBudgetsData(connector, databaseName="neo4j"):
    properties = [Property("start_date", "01-01-2021"),
                  Property("end_date", "31-12-2021")]
    for item in kg_list:
        if item:
            triple_neo4j = Triple(str(item["subject"]), item["relation"], item["object"])
            connector.AddTriple(triple_neo4j, properties)

def main():
    neoConnector = Neo4jConnector(uri="bolt://localhost:7687", user="admin", pwd="password")

    #neoConnector.remove_all()
    cypherString = CypherStringBuilder()

    test_graph = Py2NeoConnector("bolt://localhost:7687", username="admin", password="password", database="neo4j")
    
    #populateMaltaBudgetsData(test_graph)

    response_content = neoConnector.query("MATCH (n)-[r]->(m) WHERE m.name CONTAINS '100' RETURN n,r,m LIMIT 25")

    triple_set = Transformer.QueryResponseToTriple(response_content)

    print(triple_set)



if __name__ == "__main__":
    main()