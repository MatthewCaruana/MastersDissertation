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

def load_simple_questions(location, freebase_conversion_file):
    train = pd.read_csv(location + "train.txt", sep="\t", header= None)
    train = DatasetUtils.FormatSimpleQuestionsForEvidenceIntegration(train)

    valid = pd.read_csv(location + "valid.txt", sep="\t", header=None)
    valid = DatasetUtils.FormatSimpleQuestionsForEvidenceIntegration(valid)

    test = pd.read_csv(location + "test.txt", sep="\t", header=None)
    test = DatasetUtils.FormatSimpleQuestionsForEvidenceIntegration(test)

    train, valid, test = reformat_freebase(freebase_conversion_file, train, valid, test)

    return train, valid, test

def reformat_freebase(file_location, train, valid, test):
    file = open(file_location, "r", encoding="utf-8")
    conversions = {}
    for line in file:
        conversions[line[:-1].split("\t")[0][1:-1]] = line[:-1].split("\t")[2][1:-1]

    file.close()

    train_result = []
    valid_result = []
    test_result = []

    for train_single in train.values:
        if train_single[1] in conversions.keys():
            train_result.append(conversions[train_single[1]])
        else:
            train_result.append("")
    for valid_single in valid.values:
        if valid_single[1] in conversions.keys():
            valid_result.append(conversions[valid_single[1]])
        else:
            valid_result.append("")

    for test_single in test.values:
        if test_single[1] in conversions.keys():
            test_result.append(conversions[test_single[1]])
        else:
            test_result.append("")

    return train_result, valid_result, test_result

def join_data(entity_detection, relation_prediction):
    length = len(entity_detection)
    joined_data = []

    for i in range(0, length):
        joined_data.append([entity_detection[i][0], entity_detection[i][1][:-1], relation_prediction[i][1].split(".")[-1][:-1]])

    return joined_data

def get_responses(neo4jConnector, data_list, database):
    neo4jConnector.drop_graph("tempGraph", database)
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
        #betweenness = neo4jConnector.execute_betweenness("tempGraph", database)
        #degree = neo4jConnector.execute_degree_centrality("tempGraph", database)

        scored_responses = prioritize_results(responses, rank)

        neo4jConnector.drop_graph("tempGraph", database)

    return scored_responses

def prioritize_results(responses, pagerank):
    scores = []

    if responses == None or responses == []:
        return []
    else:
        best_result = [response[0].end_node for response in responses if response[0].end_node.id == pagerank[0][0]]

        return [best_result[0].id, best_result[0]["Text"], pagerank[0][2]]

    #for response in responses:
    #    pagerank_single = [p for p in pagerank if p[0] == response[0].end_node["ID"]]
    #    #pagerank_single = pagerank[response[0].end_node["ID"]]
    #    betweenness_single = betweenness[response[0].end_node["ID"]]
    #    degree_single = degree[response[0].end_node["ID"]]

    #    if len(pagerank_single) > 0:
    #        score = pagerank_single[0][2]

    #        scores.append([response[0].end_node["ID"], response[0].end_node["Text"], response[0].end_node["Type"], score])

def evaluate(actual_results, expected_results, mode):
    print("Starting Evaluation of " + mode)

    correct_match = 0
    total = len(actual_results)


    for count in range(0, len(actual_results)):
        actual_result = actual_results[count]
        expected_result = expected_results[count]

        if actual_result[1] == expected_result:
            correct_match = correct_match + 1

    accuracy = correct_match/total * 100

    return accuracy


def save_results(results, location, mode):
    print("Saving Results for " + mode)
    file = open(location + mode + ".txt", "w", encoding="utf-8")
    count = 1

    file.truncate()

    for result in results:
        file.write(mode + "-" + count + "\t" + result + "\n")

    file.close()
    print("Finished saving results for " + mode)


def save_evaluations(result, location, mode):
    print("Saving Evaluation for " + mode)
    file = open(location + mode + ".txt", "w", encoding="utf-8")

    file.write("Accuracy\t" + result)

    file.close()
    print("Finished saving results for " + mode)


def main(args):
    print(args)

    #load results from entity detection
    train_ed, test_ed, valid_ed = load_responses(args.results_entity_detection)

    #load results from relation prediction
    train_rp, test_rp, valid_rp = load_responses(args.results_relation_prediction)

    expected_train, expected_test, expected_valid = load_simple_questions(args.data_location, args.freebase_conversion_file)

    #Create querying framework
    neo4jConnector = Neo4jConnector(args.neo4j_url, args.neo4j_username, args.neo4j_password)

    train_x = join_data(train_ed, train_rp)
    test_x = join_data(test_ed, test_rp)
    valid_x = join_data(valid_ed, valid_rp)

    test_y = get_responses(neo4jConnector, test_x, args.neo4j_database)
    train_y = get_responses(neo4jConnector, train_x, args.neo4j_database)
    valid_y = get_responses(neo4jConnector, valid_x, args.neo4j_database)

    train_acc = evaluate(train_y, expected_train, "Train")
    test_acc = evaluate(test_y, expected_test, "Test")
    valid_acc = evaluate(valid_y, expected_valid, "Valid")

    save_results(train_y, args.results_evidence_integration, "Train")
    save_results(test_y, args.results_evidence_integration, "Test")
    save_results(valid_y, args.results_evidence_integration, "Valid")

    save_evaluations(train_acc, args.results_evidence_integration, "Train")
    save_evaluations(test_acc, args.results_evidence_integration, "Test")
    save_evaluations(valid_acc, args.results_evidence_integration, "Valid")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entity Detection module training/testing framework")
    parser.add_argument('--results_entity_detection', type=str, default='EntityDetection\\Results\\SimpleQuestions\\Responses\\')
    parser.add_argument('--results_evidence_integration', type=str, default='EvidenceIntegration\\Results\\SimpleQuestions\\Responses\\')
    parser.add_argument('--results_relation_prediction', type=str, default='RelationPrediction\\Results\\SimpleQuestions\\Responses\\')
    parser.add_argument('--freebase_conversion_file', type=str, default='data\\QuestionAnswering\\freebase_names\\FB5M.name.txt')
    parser.add_argument('--data_location', type=str, default='data\\QuestionAnswering\\processed_simplequestions_dataset\\')
    parser.add_argument('--neo4j_url', type=str, default="bolt://localhost:14220")
    parser.add_argument('--neo4j_username', type=str, default="matthew")
    parser.add_argument('--neo4j_password', type=str, default="password")
    parser.add_argument('--neo4j_database', type=str, default="fb2m")
    parser.add_argument('--language',type=str, default="en")
    parser.add_argument('--dataset', type=str, default="SimpleQuestions")


    args = parser.parse_args()
    main(args)