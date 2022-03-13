import argparse
import sys
import pandas as pd
import tqdm

sys.path.insert(0, "Utilities")

from RelationPredictionModule import *
from DatasetUtils import *


def load_file(file_name):
    temp = pd.read_csv(file_name)
    return temp

def evaluate(actual, predicted, mode):
    print("Starting Evaluation for " + mode)

    predicted_len = len(predicted)
    total = len(actual)
    correct = 0

    for index, predicted_row in predicted.iterrows():
        actual_row = actual.loc[actual[0] == predicted_row["Name"]] # Search by Name

        if actual_row[2].values[0] == predicted_row["EntityText"]:
            correct += 1

    if predicted_len == 0:
        precision = 0
    else:
        precision = correct / predicted_len

    if total == 0:
        recall = 0
    else:
        recall = correct / total

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

def load_simple_questions(location):
    train = pd.read_csv(location + "train.txt", sep="\t", header= None)
    train.colums = ["Name", "EntityID", "EntityText", "Relation", "ResultID", "Question", "Entities"]
    train = DatasetUtils.FormatSimpleQuestionsForEntityDetection(train)

    valid = pd.read_csv(location + "valid.txt", sep="\t", header=None)
    valid.colums = ["Name", "EntityID", "EntityText", "Relation", "ResultID", "Question", "Entities"]
    valid = DatasetUtils.FormatSimpleQuestionsForEntityDetection(valid)

    test = pd.read_csv(location + "test.txt", sep="\t", header=None)
    test.colums = ["Name", "EntityID", "EntityText", "Relation", "ResultID", "Question", "Entities"]
    test = DatasetUtils.FormatSimpleQuestionsForEntityDetection(test)

    return train, valid, test

def save_evaluations(precision, recall, f1, mode, location):
    file = open(location + mode +"_ed_results.txt", "w")
    file.write(mode)
    file.write("Precision:\t" + str(precision))
    file.write("Recall:\t" + str(recall))
    file.write("F1-score:\t" + str(f1))
    file.close()

def separate_dataset(data, dataset):
    if dataset == "SimpleQuestions":
        data_x = DatasetUtils.FormatSimpleQuestionsForQuestionOnly(data)
        data_y = DatasetUtils.FormatSimpleQuestionsForRelationOnly(data)

    return data_x, data_y

def main(args):
    print(args)
    # RELATION PREDICTION

    # Load training data
    if args.dataset == "SimpleQuestions":
        train, valid, test = load_simple_questions(args.location)

    model = RNNRelationPrediction()

    (train_x, train_y) = separate_dataset(train, args.dataset)
    (valid_x, valid_y) = separate_dataset(valid, args.dataset)
    (test_x, test_y) = separate_dataset(test, args.dataset)

    text_dictionary = DatasetUtils.dictionarise_sentences(train_x, valid_x, test_x)
    train_x = DatasetUtils.encode_sentences(text_dictionary, train_x)
    valid_x = DatasetUtils.encode_sentences(text_dictionary, valid_x)
    test_x = DatasetUtils.encode_sentences(text_dictionary, test_x)

    if args.do_training:
        model.train(train_x, train_y, test_x, test_y, args.mode, args.dataset, args.model_location)
    else:
        model.load_model(args.model_location)

    training_responses = model.detect(train, "training set", args.dataset)
    validation_responses = model.detect(valid, "valid set", args.dataset)
    testing_responses = model.detect(test, "test set", args.dataset)

    # evaluate predicted with actual
    train_precision, train_recall, train_f1 = evaluate(train, training_response, "train")
    valid_precision, valid_recall, valid_f1 = evaluate(valid, validation_responses, "valid")
    test_precision, test_recall, test_f1 = evaluate(test, testing_responses, "test")

    save_evaluations(train_precision, train_recall, train_f1, "Train", args.results_location + args.dataset + "\\")
    save_evaluations(valid_precision, valid_recall, valid_f1, "Valid", args.results_location + args.dataset + "\\")
    save_evaluations(test_precision, test_recall, test_f1, "Test", args.results_location + args.dataset + "\\")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entity Detection module training/testing framework")
    parser.add_argument('--location', type=str, default='data\\QuestionAnswering\\processed_simplequestions_dataset\\')
    parser.add_argument('--language',type=str, default="en")
    parser.add_argument('--do_training', type=bool, default=True)
    parser.add_argument('--dataset', type=str, default="SimpleQuestions")
    parser.add_argument('--model_location', type=str, default="EntityDetection\\Models\\")
    parser.add_argument('--results_location', type=str, default="EntityDetection\\Results\\")
    parser.add_argument('--mode', type=str, default="LSTM")

    args = parser.parse_args()
    main(args)