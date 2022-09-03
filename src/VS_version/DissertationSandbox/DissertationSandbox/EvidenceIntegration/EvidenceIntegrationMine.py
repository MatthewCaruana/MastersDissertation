import argparse
import sys
import pandas as pd
import tqdm

sys.path.insert(0, "Utilities")

from DatasetUtils import *
from Py2NeoConnector import *
from Neo4jConnector import *


def load_responses(file_location):
    train_results = []
    test_results = []
    valid_results = []
    
    train_file = open(file_location + "train.txt", "r", encoding="utf-8")
    for line in train_file:
        train_results.append(line.split("\t"))


    train_file.close()

    test_file = open(file_location + "test.txt", "r", encoding="utf-8")
    for line in test_file:
        test_results.append(line.split("\t"))


    test_file.close()

    valid_file = open(file_location + "valid.txt", "r", encoding="utf-8")
    for line in valid_file:
        valid_results.append(line.split("\t"))
        

    valid_file.close()

    return train_results, test_results, valid_results

def join_data(entity_detection, relation_prediction):
    length = len(entity_detection)
    joined_data = []

    for i in range(0, length):
        joined_data.append([entity_detection[i][0], entity_detection[i][1][:-1], relation_prediction[i][1].split(".")[-1][:-1]])

    return joined_data

def get_responses(neo4jConnector, data_list, database):
    responses = []
    for data in data_list:
        query_results = neo4jConnector.GetForEntityRelation(data[1], data[2], database)

        if query_results == None:
            responses.append([data[0], []])
        else:
            scored_responses = score_responses(neo4jConnector, data[1], data[2], query_results, database)
            responses.append([data[0], scored_responses])

def score_responses(neo4jConnector, root, relation, responses, database):
    scored_responses = []
    if len(responses) == 1:
        return [responses[0][0].end_node["ID"], responses[0][0].end_node["Text"], responses[0][0].end_node["Type"], 1]
    else:
        #create graph for relation and node
        neo4jConnector.create_graph_for_relation_node(relation, root, "tempGraph", database)

        rank = neo4jConnector.execute_pagerank("tempGraph", database)
        betweenness = neo4jConnector.execute_betweenness("tempGraph", database)
        degree = neo4jConnector.execute_degree_centrality("tempGraph", database)

        scored_responses = prioritize_results(responses, rank, betweenness, degree)

        neo4jConnector.drop_graph("tempGraph", database)

    return scored_responses

def prioritize_results(responses, pagerank, betweenness, degree):
    scores = []

    for response in responses:
        pagerank_single = pagerank[response[0].end_node["ID"]]
        betweenness_single = betweenness[response[0].end_node["ID"]]
        degree_single = degree[response[0].end_node["ID"]]

        score = pagerank_single

        scores.append([response[0].end_node["ID"], response[0].end_node["Text"], response[0].end_node["Type"], score])


def main(args):
    print(args)

    #load results from entity detection
    train_ed, test_ed, valid_ed = load_responses(args.results_entity_detection)

    #load results from relation prediction
    train_rp, test_rp, valid_rp = load_responses(args.results_relation_prediction)

    #Create querying framework
    neo4jConnector = Neo4jConnector(args.neo4j_url, args.neo4j_username, args.neo4j_password)

    train_x = join_data(train_ed, train_rp)
    test_x = join_data(test_ed, test_rp)
    valid_x = join_data(valid_ed, valid_rp)

    train_y = get_responses(neo4jConnector, train_x, args.neo4j_database)
    test_y = get_responses(neo4jConnector, test_x, args.neo4j_database)
    valid_y = get_responses(neo4jConnector, valid_x, args.neo4j_database)

    evaluate(train_y, "Train")
    evaluate(test_y, "Test")
    evaluate(valid_y, "Valid")

    save_results(train_y, "Train")
    save_results(test_y, "Test")
    save_results(valid_y, "Valid")

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entity Detection module training/testing framework")
    parser.add_argument('--results_entity_detection', type=str, default='EntityDetection\\Results\\SimpleQuestions\\Responses\\')
    parser.add_argument('--results_relation_prediction', type=str, default='RelationPrediction\\Results\\SimpleQuestions\\Responses\\')
    parser.add_argument('--neo4j_url', type=str, default="bolt://localhost:14220")
    parser.add_argument('--neo4j_username', type=str, default="matthew")
    parser.add_argument('--neo4j_password', type=str, default="password")
    parser.add_argument('--neo4j_database', type=str, default="fb2m")
    parser.add_argument('--language',type=str, default="en")
    parser.add_argument('--dataset', type=str, default="SimpleQuestions")


    args = parser.parse_args()
    main(args)