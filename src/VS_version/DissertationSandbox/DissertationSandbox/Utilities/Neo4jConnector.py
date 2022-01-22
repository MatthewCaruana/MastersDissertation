
from neo4j import GraphDatabase

class Neo4jConnector():
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to create the driver:", e)
        
    def close(self):
        if self.__driver is not None:
            self.__driver.close()
        
    def query(self, query, db="neo4j"):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try: 
            session = self.__driver.session(database=db) if db is not None else self.__driver.session() 
            response = list(session.run(query))
        except Exception as e:
            print("Query failed:", e)
        finally: 
            if session is not None:
                session.close()
        return response

    def execute_pagerank(self, name, db="neo4j"):
        string = "CALL gds.pageRank.stream('" + name + "') YIELD nodeId, score RETURN nodeId, gds.util.asNode(nodeId).name As name, score ORDER BY score DESC"
        return self.query(string, db)

    def execute_betweenness(self, name, db="neo4j"):
        string = "CALL gds.betweenness.stream('" + name + "') YIELD nodeId, score RETURN nodeId, gds.util.asNode(nodeId).name As name, score ORDER BY score DESC"
        return self.query(string, db)

    def execute_label_propogation(self, name, db="neo4j"):
        string = "CALL gds.labelPropogation.stream('" + name + "') YIELD nodeId, communityId As Community RETURN nodeId, gds.util.asNode(nodeId).name As Name, Community"
        return self.query(string, db)

    def execute_node_similarity(self, name, db="neo4j"):
        string = "CALL gds.nodeSimilarity.stream('" + name + "') YIELD node1, node2, similarity RETURN nodeId, gds.util.asNode(node1).name As Entity1, gds.util.asNode(node2).name As Entity2, similarity ORDER BY similarity DESC"
        return self.query(string, db)

    def get_related_to_node(self, name, db="neo4j"):
        string = "MATCH p=(n:`" + name + "`)-->() RETURN p"
        results = self.query(string, db)

        triples = []
        for result in results:
            values = result.values()[0]
            relations = values.relationships
            for relation in relations:
                triples.append(Triple(relation.start_node['name'], relation.type, relation.end_node['name']).convert_to_json())

        return triples

    def get_related_to_relation(self, name, db="neo4j"):
        string = "MATCH p=()-[r:`" + name +"`]->() RETURN p"
        results = self.query(string, db)

        triples = []
        for result in results:
            values = result.values()[0]
            relations = values.relationships
            for relation in relations:
                triples.append(Triple(relation.start_node['name'], relation.type, relation.end_node['name']).convert_to_json())

        return triples

    def create_graph_for_node(self, node, name, db="neo4j"):
        string = "CALL gds.graph.create('" + name + "','" \
            + node + "', '" \
            + "'*')"
        self.query(string, db)
        print("Graph Created")

    def create_graph_for_relation_node(self, relation, node, name, db="neo4j"):
        string = "CALL gds.graph.create('" + name + "','" \
            + node + "', '" \
            + relation + "')"
        self.query(string, db)
        print("Graph Created")

    #['CD', 'JJ', 'JJS', 'NN','NNP', 'NNS', 'RB', 'VB', 'VBD', 'VBG', 'VBN']
    def create_graph_for_relation(self, relation, name, db="neo4j"):
        string = "CALL gds.graph.create('" + name + "'," \
            + "'*', '" \
            + relation + "')"
        self.query(string, db)
        print("Graph Created")

    def drop_graph(self, name, db="neo4j"):
        string = "CALL gds.graph.drop('" + name + "') YIELD graphName;"
        self.query(string, db)
        print("Graph Removed")

    def remove_all(self, db="neo4j"):
        remove_related_nodes = "match (a) -[r] -> () delete a, r"
        self.query(remove_related_nodes, db)

        remove_unrelated_nodes = "match (a) delete a"
        self.query(remove_unrelated_nodes, db)
        print("Database Cleared")

    def list_graph(self, name, db="neo4j"):
        string = "CALL gds.graph.list('" + name + "')"
        return self.query(string, db)