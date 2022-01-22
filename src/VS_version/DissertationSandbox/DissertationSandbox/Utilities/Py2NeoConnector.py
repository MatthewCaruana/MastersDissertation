import sys

sys.path.insert(0, "Utilities")

from Triples import *
from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher
from py2neo.bulk import create_nodes, create_relationships

class Py2NeoConnector:
    def __init__(self, uri, username= "admin", password="password", database="neo4j"):
        self.graph = Graph("bolt://localhost:7687", name=database, user=username, password=password)
        self.matcher = NodeMatcher(self.graph)

    def AddTriple(self, triple, properties):
        # Create Subject
        node_subject = Node(triple.subject, name=triple.subject)
        try:
            self.graph.merge(node_subject, triple.subject,"name")
            for property in properties:
                node_subject[property.Name] = property.Text
            node_subject["RelationType"] = "Subject"
            self.graph.push(node_subject)
        except:
            print("Something went wrong in subject")

        # Create Object
        node_object = Node(triple.object, name=triple.object)
        try:
            self.graph.merge(node_object,triple.object,"name")
            for property in properties:
                node_object[property.Name] = property.Text
            node_subject["RelationType"] = "Object"
            self.graph.push(node_object)
        except:
            print("Something went wrong in object")

        # Create Relation
        relationship = Relationship(node_subject, triple.relation, node_object)
        try:
            self.graph.merge(relationship,triple.relation,"name")
            node_subject["RelationType"] = "Relation"
            self.graph.push(relationship)
        except:
            print("Something went wrong in rel")


    def AddNodes(self, nodes, keys):
        node_list = [[node] for node in nodes]
        
        chunks = [node_list[x:x+1000] for x in range(0, len(node_list), 1000)]

        for chunk in chunks:
            create_nodes(self.graph.auto(), chunk, ("Entities", "Names"), keys)
        
        print(self.graph.nodes.match("Names").count())

    def AddRelationsWithCypher(self, relation_list, relation_type):
        if len(relation_list) > 100:
            chunks = [relation_list[x:x+100] for x in range(0,len(relation_list), 100)]
        else:
            chunks = [relation_list]

        i = 0
        for chunk in chunks:
            tx = self.graph.begin()
            print("Creating Relations: " + str(i) + "/" + str(len(chunks)-1))
            for relationship in chunk:
                relation_string = "MATCH (start:Entities), (end:Entities) WHERE start.name = \"{}\" AND end.name= \"{}\" CREATE (start)-[r:{}]->(end)".format(relationship[0], relationship[2],relation_type)
                tx.run(relation_string)

            tx.commit()
            i = i + 1

    def AddRelationsAndNodesWithCypher(self, relation_list, relation_type):
        if len(relation_list) > 100:
            chunks = [relation_list[x:x+100] for x in range(0,len(relation_list), 100)]
        else:
            chunks = [relation_list]

        i = 0
        for chunk in chunks:
            tx = self.graph.begin()
            print("Creating Relations: " + str(i) + "/" + str(len(chunks)-1))
            for relationship in chunk:
                relation_string = "MERGE (n:Entities {{name: \"{}\"}}) MERGE (m:Entities {{name: \"{}\"}}) MERGE (n)-[:{}]->(m)".format(relationship[0], relationship[2],relation_type)
                tx.run(relation_string)

            tx.commit()
            i = i + 1

    def AddRelations(self, relation_list, relation_type):
        if len(relation_list) > 100:
            chunks = [relation_list[x:x+100] for x in range(0,len(relation_list), 100)]
        else:
            chunks = [relation_list]

        i = 0
        for chunk in chunks:
            print("Creating Relations: " + str(i) + "/" + str(len(chunks)))
            create_relationships(self.graph.auto(), chunk, relation_type, start_node_key=("Entities", "name"), end_node_key=("Entities", "name"))
            i = i + 1


    def AddRelationWithNodes(self, sub, rel, obj):
        noNode = False

        subjectNode = self.matcher.match("Entities").where(name=sub).first()
        if subjectNode == None or subjectNode is None:
            noNode = True

        objectNode = self.matcher.match("Entities").where(name=obj).first()
        if objectNode == None or objectNode is None:
            noNode = True

        if noNode == False:
            relation = Relationship(subjectNode, rel, objectNode)

            self.graph.create(relation)
        else:
            print("Skipped following relation: (" + sub + ", " + rel + ", " + obj + ")")

    def RelationExists(self, relation, relations_total):
        count = len(RelationshipMatcher(self.graph).match((None, None), relation))

        if count == relations_total:
            return True
        else:
            return False

    def LoadCSV(self, csv_location):
        tx = self.graph.begin()
        load_csv_string = "LOAD CSV WITH HEADERES FROM"