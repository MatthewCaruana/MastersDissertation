import argparse
import sys
import pandas as pd
import tqdm
import numpy as np

sys.path.insert(0, "Utilities")

from RelationPredictionModule import *
from DatasetUtils import *


def load_file(file_name):
    temp = pd.read_csv(file_name)
    return temp

def evaluate(actual, predicted, mode, label_size):
    print("Starting Evaluation for " + mode)

    predicted_len = len(predicted)
    total = len(actual)

    count = 0
    top5_indices_predicted = [np.argsort(predicted_single)[-5:] for predicted_single in predicted]

    predicted_normalized = predicted.argmax(axis=0)
    new_predicted = []
    new_predicted_top_5 = []
    for row in predicted:

        highest_index = row.argmax(axis=0)

        new_token = [0]*label_size
        new_token[highest_index] = 1

        new_predicted.append(new_token)

    for top5_indices in top5_indices_predicted:
        row_prediction = []
        for row in top5_indices:
            new_token= [0]*label_size
            new_token[row] = 1

            row_prediction.append(new_token)

        new_predicted_top_5.append(row_prediction)

    count = 0
    correct = 0
    incorrect = 0

    correct_5 = 0
    incorrect_5 = 0
    for predicted_row in new_predicted:
        actual_row = actual[count] # Search by Name

        if actual_row == predicted_row:
            correct += 1
        else:
            incorrect += 1 

        count += 1

    count = 0
    for predicted_rows_top5 in new_predicted_top_5:
        actual_row = actual[count]
        if actual_row in predicted_rows_top5:
            correct_5 += 1
        else:
            incorrect_5 += 1

        count += 1

    accuracy = correct / (correct + incorrect)
    accuracy_top_5 = correct_5 / (correct_5 + incorrect_5)

    print("Accuracy: " + str(accuracy * 100))
    print("Accuracy Top 5: " + str(accuracy_top_5 * 100))

    return accuracy, accuracy_top_5

def load_simple_questions(location):
    train = pd.read_csv(location + "train.txt", sep="\t", header= None)
    #train.colums = ["Name", "EntityID", "EntityText", "Relation", "ResultID", "Question", "Entities"]
    train = DatasetUtils.FormatSimpleQuestionsForRelationPrediction(train)

    valid = pd.read_csv(location + "valid.txt", sep="\t", header=None)
    #valid.colums = ["Name", "EntityID", "EntityText", "Relation", "ResultID", "Question", "Entities"]
    valid = DatasetUtils.FormatSimpleQuestionsForRelationPrediction(valid)

    test = pd.read_csv(location + "test.txt", sep="\t", header=None)
    #test.colums = ["Name", "EntityID", "EntityText", "Relation", "ResultID", "Question", "Entities"]
    test = DatasetUtils.FormatSimpleQuestionsForRelationPrediction(test)

    return train, valid, test

def load_doi_questions(location):
    train = pd.read_csv(location + "train.txt", sep="\t", header= None)
    #train.colums = ["Name", "EntityID", "EntityText", "Relation", "ResultID", "Question", "Entities"]
    train = DatasetUtils.FormatDOIForRelationPrediction(train)

    valid = pd.read_csv(location + "valid.txt", sep="\t", header=None)
    #valid.colums = ["Name", "EntityID", "EntityText", "Relation", "ResultID", "Question", "Entities"]
    valid = DatasetUtils.FormatDOIForRelationPrediction(valid)

    test = pd.read_csv(location + "test.txt", sep="\t", header=None)
    #test.colums = ["Name", "EntityID", "EntityText", "Relation", "ResultID", "Question", "Entities"]
    test = DatasetUtils.FormatDOIForRelationPrediction(test)

    return train, valid, test

def save_responses(original_relations, responses, mode, location):
    print("Starting response gathering for " + mode)

    predicted_len = len(responses)
    top5_indices_predicted = [np.argsort(predicted_single)[-5:] for predicted_single in responses]

    count = 0
    new_predicted = []
    for row in responses:
        new_predicted.append(row.argmax())

    file = open(location + mode + ".txt", "w+", encoding="utf-8")
    count = 1
    for predicted_index in new_predicted:
        file.write(mode + "-" + str(count) + "\t")

        file.write(original_relations[predicted_index+1])
        file.write("\n")
        count += 1
    file.close()

    file = open(location + mode + "_top_5.txt", "w+", encoding="utf-8")
    count = 0
    for predicted_rows_top5 in top5_indices_predicted:
        file.write(mode + "-" + str(count) + "\t")

        predictions = []
        for prediction_index in predicted_rows_top5:
            predictions.append(original_relations[prediction_index+1])

        file.write(str(predictions))
        file.write("\n")

        count += 1
    
    file.close()
    print("Response gathering for " + mode + " completed")

def save_evaluations(accuracy, accuracy_top_5, mode, location):
    file = open(location + mode +"_rp_results.txt", "w")
    file.write(mode)
    file.write("\nAccuracy:\t" + str(accuracy))
    file.write("\nAccuracy Top 5:\t" + str(accuracy_top_5))
    file.close()

def separate_dataset(data, dataset):
    if dataset == "SimpleQuestions":
        data_x = DatasetUtils.FormatSimpleQuestionsForQuestionOnly(data)
        data_y = DatasetUtils.FormatSimpleQuestionsForRelationOnly(data)
    elif dataset == "DOI":
        data_x = DatasetUtils.FormatDOIForQuestionOnly(data)
        data_y = DatasetUtils.FormatDOIForRelationOnly(data)

    return data_x, data_y

def main(args):
    print(args)
    # RELATION PREDICTION

    # Load training data
    if args.dataset == "SimpleQuestions":
        train, valid, test = load_simple_questions(args.location)
    elif args.dataset == "DOI":
        train, valid, test = load_doi_questions(args.location)


    (train_x, train_y) = separate_dataset(train, args.dataset)
    (valid_x, valid_y) = separate_dataset(valid, args.dataset)
    (test_x, test_y) = separate_dataset(test, args.dataset)

    text_dictionary = DatasetUtils.dictionarise_sentences(train_x, valid_x, test_x)
    relation_dictionary = DatasetUtils.dictionarise_relations(train_y, valid_y, test_y)
    inverse_dictionary = {index: token for token, index in relation_dictionary.items()}

    model = RNNRelationPrediction(len(relation_dictionary), len(text_dictionary))

    train_x = DatasetUtils.encode_sentences(text_dictionary, train_x)
    valid_x = DatasetUtils.encode_sentences(text_dictionary, valid_x)
    test_x = DatasetUtils.encode_sentences(text_dictionary, test_x)

    train_y = DatasetUtils.encode_relations(train_y, relation_dictionary)
    valid_y = DatasetUtils.encode_relations(valid_y, relation_dictionary)
    test_y = DatasetUtils.encode_relations(test_y, relation_dictionary)

    if args.do_training:
        model.train(train_x, train_y, valid_x, valid_y, args.mode, args.dataset, args.model_location + args.model_name)
    else:
        model.load_model(args.model_location + args.model_name + ".h5")

    #train_response_single = model.predict_single(train_x[1])

    training_responses = model.detect(train_x, "training set", args.dataset)
    validation_responses = model.detect(valid_x, "valid set", args.dataset)
    testing_responses = model.detect(test_x, "test set", args.dataset)

    # evaluate predicted with actual
    train_accuracy, train_accuracy_top_5 = evaluate(train_y, training_responses, "train", len(relation_dictionary))
    valid_accuracy, valid_accuracy_top_5 = evaluate(valid_y, validation_responses, "valid", len(relation_dictionary))
    test_accuracy, test_accuracy_top_5 = evaluate(test_y, testing_responses, "test", len(relation_dictionary))

    save_evaluations(train_accuracy, train_accuracy_top_5, "Train", args.results_location + args.dataset + "\\")
    save_evaluations(valid_accuracy, valid_accuracy_top_5, "Valid", args.results_location + args.dataset + "\\")
    save_evaluations(test_accuracy, test_accuracy_top_5, "Test", args.results_location + args.dataset + "\\")

    train_y = DatasetUtils.decode_relations(inverse_dictionary, train_y)
    valid_y = DatasetUtils.decode_relations(inverse_dictionary, valid_y)
    test_y = DatasetUtils.decode_relations(inverse_dictionary, test_y)

    save_responses(inverse_dictionary, training_responses, "train", args.results_location + args.dataset + "\\Responses\\")
    save_responses(inverse_dictionary, testing_responses, "test", args.results_location + args.dataset + "\\Responses\\")
    save_responses(inverse_dictionary, validation_responses, "valid", args.results_location + args.dataset + "\\Responses\\")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relation Prediction module training/testing framework")
    #parser.add_argument('--location', type=str, default='data\\QuestionAnswering\\processed_simplequestions_dataset\\')
    parser.add_argument('--location', type=str, default='data\\DOI\\QA\\maltese\\')
    parser.add_argument('--language',type=str, default="en")
    parser.add_argument('--do_training', type=bool, default=True)
    parser.add_argument('--dataset', type=str, default="DOI")
    parser.add_argument('--model_location', type=str, default="RelationPrediction\\Models\\")
    parser.add_argument('--model_name', type=str, default="doi_model_mt")
    parser.add_argument('--results_location', type=str, default="RelationPrediction\\Results\\")
    parser.add_argument('--mode', type=str, default="LSTM")

    args = parser.parse_args()
    main(args)