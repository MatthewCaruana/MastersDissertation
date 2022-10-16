import argparse
import sys
import pandas as pd
from tqdm import tqdm
import ast
from nltk import ngrams

sys.path.insert(0, "Utilities")

from DatasetUtils import *
from Py2NeoConnector import *
from Neo4jConnector import *


def load_responses(file_location, top_5_files):
    train_results = []
    test_results = []
    valid_results = []
    
    if top_5_files == False:
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
    else:
        train_file = open(file_location + "train_top_5.txt", "r", encoding="utf-8")
        for line in train_file:
            train_results.append(line.split("\t"))

        train_file.close()

        test_file = open(file_location + "test_top_5.txt", "r", encoding="utf-8")
        for line in test_file:
            test_results.append(line.split("\t"))

        test_file.close()

        valid_file = open(file_location + "valid_top_5.txt", "r", encoding="utf-8")
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

def load_doi(location):
    train = pd.read_csv(location + "train.txt", sep="\t", header= None)
    train = DatasetUtils.FormatDOIForEvidenceIntegration(train)

    valid = pd.read_csv(location + "valid.txt", sep="\t", header=None)
    valid = DatasetUtils.FormatDOIForEvidenceIntegration(valid)

    test = pd.read_csv(location + "test.txt", sep="\t", header=None)
    test = DatasetUtils.FormatDOIForEvidenceIntegration(test)

    train_results = [train_single[1] for train_single in train.values]
    test_results = [test_single[1] for test_single in test.values]
    valid_results = [valid_single[1] for valid_single in valid.values]

    return train_results, valid_results, test_results

def reformat_freebase(file_location, train, valid, test):
    file = open(file_location, "r", encoding="utf-8")
    conversions = {}
    for line in file:
        if line[:-1].split("\t")[0][1:-1] in conversions.keys():
            conversions[line[:-1].split("\t")[0][1:-1]].append(line[:-1].split("\t")[2][1:-1])
        else:
            conversions[line[:-1].split("\t")[0][1:-1]] = [line[:-1].split("\t")[2][1:-1]]

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

def join_data(entity_detection, relation_prediction, rp_top_5):
    length = len(entity_detection)
    joined_data = []

    for i in range(0, length):
        rp_top_5_single = ast.literal_eval(rp_top_5[i][1])
        rp_top_5_single = [rp_single.split(".")[-1] for rp_single in rp_top_5_single]

        joined_data.append([entity_detection[i][0], entity_detection[i][1][:-1], relation_prediction[i][1].split(".")[-1][:-1], rp_top_5_single])

    return joined_data

def get_responses(neo4jConnector, data_list, database):
    neo4jConnector.drop_graph("tempGraph", database)

    responses = []
    responses_top_5 = []

    for data in tqdm(data_list):
        query_results = neo4jConnector.GetForEntityRelation(data[1], data[2], database)

        if database == "fb2m":
            if query_results == [] or query_results == None:
                responses.append([data[0], [], []])
            else:
                all_responses, scored_responses = score_responses(neo4jConnector, data[1], data[2], query_results, database)
                responses.append([data[0], scored_responses, all_responses])

            for relation in data[3]:
                query_results = neo4jConnector.GetForEntityRelation(data[1], relation, database)

                response = []

                if query_results == [] or query_results == None:
                    response.append([[], []])
                else:
                    all_responses, scored_responses = score_responses(neo4jConnector, data[1], relation, query_results, database)
                    response.append([scored_responses, all_responses])

            responses_top_5.append([data[0], response])

        elif database == "doi-en" or database == "doi-mt" or database == "doi-mt2":
            if query_results == [] or query_results == None:
                #generate n-grams of max 3 characters
                all_responses = []
                scored_responses = []
                if len(data[1].split()) > 1:
                    ngram_entities = ngrams(data[1].split(), len(data[1].split()))
                    for entity_grouping in ngram_entities:
                        for entity in entity_grouping:
                            query_results = neo4jConnector.GetForEntityRelation(entity, data[2], database)

                            if query_results == [] or query_results == None:
                                #responses.append([data[0], [], []])
                                print("Hi")
                            else:
                                all_responses_ngram, scored_responses_ngram = score_responses(neo4jConnector, entity, data[2], query_results, database)
                                all_responses.extend(all_responses_ngram)
                                scored_responses.extend(scored_responses_ngram)
                responses.append([data[0], scored_responses, all_responses])
            else:
                all_responses, scored_responses = score_responses(neo4jConnector, data[1], data[2], query_results, database)
                responses.append([data[0], scored_responses, all_responses])

    return responses, responses_top_5

def score_responses(neo4jConnector, root, relation, responses, database):
    scored_responses = []
    if len(responses) == 1:
        return [responses[0][0].end_node["ID"], responses[0][0].end_node["Text"]],[responses[0][0].end_node["ID"], responses[0][0].end_node["Text"], responses[0][0].end_node["Type"], 1]
    else:
        #create graph for relation and node
        neo4jConnector.create_graph_for_relation_node(relation, root, "tempGraph", database)

        rank = neo4jConnector.execute_pagerank("tempGraph", database)
        #betweenness = neo4jConnector.execute_betweenness("tempGraph", database)
        #degree = neo4jConnector.execute_degree_centrality("tempGraph", database)
        if rank == None:
            all_responses, scored_responses = allocate_results(responses)
        else:
            all_responses, scored_responses = prioritize_results(responses, rank)

        neo4jConnector.drop_graph("tempGraph", database)

    return all_responses, scored_responses

def allocate_results(responses):
    all_responses = []
    for response in responses:
        all_responses.append([response[0].end_node.id, response[0].end_node["Text"]])

    return all_responses, []

def prioritize_results(responses, pagerank):
    scores = []

    if responses == None or responses == []:
        return [], []
    else:
        if pagerank == None or pagerank == []:
            return [],[]
        else:
            all_results = [[response[0].end_node.id, response[0].end_node["Text"]] for response in responses]
            best_result = [response[0].end_node for response in responses if response[0].end_node.id == pagerank[0][0]]

            if best_result == []:
                return all_results, []
            else:
                return all_results, [best_result[0].id, best_result[0]["Text"], pagerank[0][2]]

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
    nothing_found = 0
    found_count = 0
    total = len(actual_results)


    for count in range(0, len(actual_results)):
        actual_result = actual_results[count]
        expected_result = expected_results[count]

        if actual_result[1] == []:
            nothing_found = nothing_found + 1
        else:
            query_text = actual_result[1][1]
            if query_text != None:
                if any(actual_result[1][1] in x for x in expected_result) or actual_result[1][1] == expected_result:
                    correct_match = correct_match + 1
            
            if actual_result[2] != None:
                if isinstance(actual_result[2][0], list):
                    for all_result_single in actual_result[2]:
                        if all_result_single[1] != None:
                            if any(all_result_single[1] in x for x in expected_result) or all_result_single[1] == expected_result:
                                found_count = found_count + 1
                                break
                elif isinstance(actual_result[2][0], str):
                    if actual_result[2][1] != None:
                        if any(actual_result[2][1] in x for x in expected_result):
                            found_count = found_count + 1

    accuracy = correct_match/total * 100
    found_accuracy = found_count/total * 100

    print("Accuracy:\t" + str(accuracy))
    print("Found Accuracy:\t" + str(found_accuracy))

    return accuracy, found_accuracy, nothing_found

def evaluateTop5(actual_results, expected_results, mode):
    print("Starting Evaluation of " + mode)

    correct_match = 0
    nothing_found = 0
    found_count = 0
    total = len(actual_results)


    for count in range(0, len(actual_results)):
        actual_result = actual_results[count]
        expected_result = expected_results[count]

        all_actual = []
        all_actual_scored = []

        for row in actual_result[1]:
            if row[0] != []:
                all_actual_scored.append(row[0][1])
            if row[1] != []:
                if isinstance(row[1][0], list):
                    all_actual.append([element[1] for element in row[1]])
                elif isinstance(row[1][0], str):
                    all_actual.append(row[1][1])

        if len(expected_result) > 1:
            if any(item in expected_result for item in  all_actual):
                found_count += 1
            else:
                nothing_found += 1
            if any(item in expected_result for item in  all_actual_scored):
                correct_match += 1
        else:
            if expected_result[0] in all_actual:
                found_count += 1
            else:
                nothing_found += 1
            if expected_result[0] in all_actual_scored:
                correct_match += 1

    accuracy = correct_match/total * 100
    found_accuracy = found_count/total * 100

    print("Accuracy:\t" + str(accuracy))
    print("Found Accuracy:\t" + str(found_accuracy))

    return accuracy, found_accuracy, nothing_found


def save_results(results, location, mode):
    print("Saving Results for " + mode)
    file = open(location + mode + ".txt", "w+", encoding="utf-8")

    file.truncate()

    for result in results:
        file.write(result[0] + "\t" + str(result[1]) + "\t" + str(result[2]) + "\n")

    file.close()
    print("Finished saving results for " + mode)

def save_results_top_5(results, location, mode):
    print("Saving Results for " + mode)
    file = open(location + mode + ".txt", "w+", encoding="utf-8")

    file.truncate()

    for result in results:
        file.write(result[0] + "\t" + str(result[1]) + "\n")

    file.close()
    print("Finished saving results for " + mode)

def save_evaluations(result, location, mode):
    print("Saving Evaluation for " + mode)
    file = open(location + mode + ".txt", "w", encoding="utf-8")

    file.write("Accuracy\t" + str(result))

    file.close()
    print("Finished saving results for " + mode)

def get_responses_saved(location, mode):
    data = []

    print("Loading Responses for " + mode)
    file = open(location + mode + ".txt", 'r', encoding="utf-8")

    for line in file:
        data.append([line.split("\t")[0], ast.literal_eval(line.split("\t")[1]), ast.literal_eval(line.split("\t")[2])])
    
    file.close()
    return data


def main(args):
    print(args)

    #load results from entity detection
    train_ed, test_ed, valid_ed = load_responses(args.results_entity_detection, False)

    #load results from relation prediction
    train_rp, test_rp, valid_rp = load_responses(args.results_relation_prediction, False)

    train_rp_top_5, test_rp_top_5, valid_rp_top_5 = load_responses(args.results_relation_prediction, True)

    if (args.dataset == "SimpleQuestions"):
        expected_train, expected_valid, expected_test = load_simple_questions(args.data_location, args.freebase_conversion_file)
    elif(args.dataset == "DOI"):
        expected_train, expected_valid, expected_test = load_doi(args.data_location)

    #Create querying framework
    neo4jConnector = Neo4jConnector(args.neo4j_url, args.neo4j_username, args.neo4j_password)

    train_x = join_data(train_ed, train_rp, train_rp_top_5)
    test_x = join_data(test_ed, test_rp, test_rp_top_5)
    valid_x = join_data(valid_ed, valid_rp, valid_rp_top_5)

    if(args.skip == False):
        test_y, test_y_top_5 = get_responses(neo4jConnector, test_x, args.neo4j_database)
        train_y, train_y_top_5 = get_responses(neo4jConnector, train_x, args.neo4j_database)
        valid_y, valid_y_top_5 = get_responses(neo4jConnector, valid_x, args.neo4j_database)

        save_results(train_y, args.results_evidence_integration, "Train")
        save_results(test_y, args.results_evidence_integration, "Test")
        save_results(valid_y, args.results_evidence_integration, "Valid")

        save_results_top_5(train_y_top_5, args.results_evidence_integration, "Train_top5")
        save_results_top_5(test_y_top_5, args.results_evidence_integration, "Test_top5")
        save_results_top_5(valid_y_top_5, args.results_evidence_integration, "Valid_top5")
    else:
        train_y = get_responses_saved(args.results_evidence_integration, "Train")
        test_y = get_responses_saved(args.results_evidence_integration, "Test")
        valid_y = get_responses_saved(args.results_evidence_integration, "Valid")

    train_acc, train_found_acc, train_nothing_found = evaluate(train_y, expected_train, "Train")
    test_acc, test_found_acc, test_nothing_found = evaluate(test_y, expected_test, "Test")
    valid_acc, valid_found_acc, valid_nothing_found = evaluate(valid_y, expected_valid, "Valid")

    train_acc_top_5, train_found_acc_top_5, train_nothing_found_top_5 = evaluateTop5(train_y_top_5, expected_train, "Train")
    test_acc_top_5, test_found_acc_top_5, test_nothing_found_top_5 = evaluateTop5(test_y_top_5, expected_test, "Test")
    valid_acc_top_5, valid_found_acc_top_5, valid_nothing_found_top_5 = evaluateTop5(valid_y_top_5, expected_valid, "Valid")

    save_evaluations(train_acc, args.results_evidence_integration_stats, "Train")
    save_evaluations(test_acc, args.results_evidence_integration_stats, "Test")
    save_evaluations(valid_acc, args.results_evidence_integration_stats, "Valid")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entity Detection module training/testing framework")
    #parser.add_argument('--results_entity_detection', type=str, default='EntityDetection\\Results\\SimpleQuestions\\Responses\\')
    #parser.add_argument('--results_evidence_integration', type=str, default='EvidenceIntegration\\Results\\SimpleQuestions\\Responses\\')
    #parser.add_argument('--results_evidence_integration_stats', type=str, default='EvidenceIntegration\\Results\\SimpleQuestions\\')
    #parser.add_argument('--results_relation_prediction', type=str, default='RelationPrediction\\Results\\SimpleQuestions\\Responses\\')
    #parser.add_argument('--data_location', type=str, default='data\\QuestionAnswering\\processed_simplequestions_dataset\\')
    parser.add_argument('--results_entity_detection', type=str, default='EntityDetection\\Results\\DOI-mt2\\Responses\\')
    parser.add_argument('--results_evidence_integration', type=str, default='EvidenceIntegration\\Results\\DOI-mt2\\Responses\\')
    parser.add_argument('--results_evidence_integration_stats', type=str, default='EvidenceIntegration\\Results\\DOI-mt2\\')
    parser.add_argument('--results_relation_prediction', type=str, default='RelationPrediction\\Results\\DOI-mt2\\Responses\\')
    parser.add_argument('--data_location', type=str, default='data\\DOI\\QA\\maltese\\')
    parser.add_argument('--freebase_conversion_file', type=str, default='data\\QuestionAnswering\\freebase_names\\FB5M.name.txt')
    parser.add_argument('--neo4j_url', type=str, default="bolt://localhost:14220")
    parser.add_argument('--neo4j_username', type=str, default="matthew")
    parser.add_argument('--neo4j_password', type=str, default="password")
    parser.add_argument('--neo4j_database', type=str, default="doi-mt2")
    parser.add_argument('--language',type=str, default="en")
    parser.add_argument('--dataset', type=str, default="DOI")
    parser.add_argument('--skip', type=bool, default=False)


    args = parser.parse_args()
    main(args)