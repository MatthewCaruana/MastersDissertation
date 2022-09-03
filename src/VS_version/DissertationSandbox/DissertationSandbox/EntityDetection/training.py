import argparse
import sys
import pandas as pd
import tqdm

sys.path.insert(0, "Utilities")

from EntityDetectionModule import *
from DatasetUtils import *

def load_file(file_name):
    temp = pd.read_csv(file_name)
    return temp

def evaluate(actual, predicted, mode):
    print("Starting Evaluation for " + mode)

    predicted_len = len(predicted)
    total = len(actual)
    correct = 0

    count = 0
    new_predicted = []
    for row in predicted:
        new_row = []
        for token in row:
            new_token = []
            if token[0] > token[1] or (token[0] < 0 and token[1] < 0):
                new_token = [1,0]
            else:
                new_token = [0,1]
            new_row.append(new_token)
        new_predicted.append(new_row)

    count = 0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for predicted_row in new_predicted:
        actual_row = actual[count] # Search by Name

        for i in range(0, 299):
            if actual_row[i][0] == 0 and actual_row[i][1] == 1:
                if actual_row[i] == predicted_row[i]:
                    true_pos += 1
                else:
                    false_pos += 1
            else:
                if actual_row[i] == predicted_row[i]:
                    true_neg += 1
                else:
                    false_neg += 1

        if actual_row == predicted_row:
            correct += 1

        count += 1

    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    print("Accuracy: " + str(accuracy * 100))
    print("Precision: " + str(precision * 100))
    print("Recall: " + str(recall * 100))
    print("F-Measure: " + str(f1 * 100))

    return precision, recall, f1

def load_simple_questions(location):
    train = pd.read_csv(location + "train.txt", sep="\t", header= None)
    train = DatasetUtils.FormatSimpleQuestionsForEntityDetection(train)

    valid = pd.read_csv(location + "valid.txt", sep="\t", header=None)
    valid = DatasetUtils.FormatSimpleQuestionsForEntityDetection(valid)

    test = pd.read_csv(location + "test.txt", sep="\t", header=None)
    test = DatasetUtils.FormatSimpleQuestionsForEntityDetection(test)

    return train, valid, test

def separate_dataset(data, dataset):
    if dataset == "SimpleQuestions":
        data_x = DatasetUtils.FormatSimpleQuestionsForQuestionOnly(data)
        data_y = DatasetUtils.FormatSimpleQuestionsForEntitiesOnly(data)

    return data_x, data_y

def save_evaluations(precision, recall, f1, mode, location):
    file = open(location + mode +"_ed_results.txt", "w")
    file.write(mode)
    file.write("Precision:\t" + str(precision))
    file.write("Recall:\t" + str(recall))
    file.write("F1-score:\t" + str(f1))
    file.close()

def save_responses(original_sentences, responses, mode, location):
    print("Starting response gathering for " + mode)

    predicted_len = len(responses)

    count = 0
    new_predicted = []
    for row in responses:
        new_row = []
        for token in row:
            new_token = []
            if token[0] > token[1] or (token[0] < 0 and token[1] < 0):
                new_token = [1,0]
            else:
                new_token = [0,1]
            new_row.append(new_token)
        new_predicted.append(new_row)

    file = open(location + mode + ".txt", "w+", encoding="utf-8")
    file.write(mode + ":\n")
    count = 1
    for row in new_predicted:
        file.write(mode + "-" + str(count) + "\t")
        row_entities = []
        token_count = 0
        for token in row:
            if token == [0, 1]:
                if token_count < len(original_sentences[count-1]):
                    row_entities.append(original_sentences[count-1][token_count])
            token_count += 1

        if len(row_entities) > 1:
            file.write(" ".join(row_entities))
        elif len(row_entities) == 1:
            file.write(row_entities[0])
        file.write("\n")
        count += 1
    
    file.close()
    print("Response gathering for " + mode + " completed")

def main(args):
    print(args)
    # ENTITY DETECTION

    # Load training data
    if args.dataset == "SimpleQuestions":
        train, valid, test = load_simple_questions(args.location)

    (train_x, train_y) = separate_dataset(train, args.dataset)
    (valid_x, valid_y) = separate_dataset(valid, args.dataset)
    (test_x, test_y) = separate_dataset(test, args.dataset)

    text_dictionary = DatasetUtils.dictionarise_sentences(train_x, valid_x, test_x)
    inverse_dictionary = {index: token for token, index in text_dictionary.items()}

    model = NeuralEntityDetection(len(inverse_dictionary))

    train_x = DatasetUtils.encode_sentences(text_dictionary, train_x)
    valid_x = DatasetUtils.encode_sentences(text_dictionary, valid_x)
    test_x = DatasetUtils.encode_sentences(text_dictionary, test_x)

    train_y = DatasetUtils.encode_entities(train_y)
    valid_y = DatasetUtils.encode_entities(valid_y)
    test_y = DatasetUtils.encode_entities(test_y)

    if args.do_training:
        model.train(train_x, train_y, test_x, test_y, args.mode, args.dataset, args.model_location + args.model_name)
    else:
        model.load_model(args.model_location + args.model_name)

    training_responses = model.detect(train_x, "training set", args.dataset)
    validation_responses = model.detect(valid_x, "valid set", args.dataset)
    testing_responses = model.detect(test_x, "test set", args.dataset)

    # evaluate predicted with actual
    train_precision, train_recall, train_f1 = evaluate(train_y, training_responses, "train")
    valid_precision, valid_recall, valid_f1 = evaluate(valid_y, validation_responses, "valid")
    test_precision, test_recall, test_f1 = evaluate(test_y, testing_responses, "test")

    save_evaluations(train_precision, train_recall, train_f1, "Train", args.results_location + args.dataset + "\\")
    save_evaluations(valid_precision, valid_recall, valid_f1, "Valid", args.results_location + args.dataset + "\\")
    save_evaluations(test_precision, test_recall, test_f1, "Test", args.results_location + args.dataset + "\\")

    train_x = DatasetUtils.decode_sentences(inverse_dictionary, train_x)
    valid_x = DatasetUtils.decode_sentences(inverse_dictionary, valid_x)
    test_x = DatasetUtils.decode_sentences(inverse_dictionary, test_x)

    save_responses(train_x, training_responses, "train", args.results_location + args.dataset + "\\Responses\\")
    save_responses(test_x, testing_responses, "test", args.results_location + args.dataset + "\\Responses\\")
    save_responses(valid_x, validation_responses, "valid", args.results_location + args.dataset + "\\Responses\\")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entity Detection module training/testing framework")
    parser.add_argument('--location', type=str, default='data\\QuestionAnswering\\processed_simplequestions_dataset\\')
    parser.add_argument('--language',type=str, default="en")
    parser.add_argument('--do_training', type=bool, default=False)
    parser.add_argument('--dataset', type=str, default="SimpleQuestions")
    parser.add_argument('--model_location', type=str, default="EntityDetection\\Models\\")
    parser.add_argument('--model_name', type=str, default="simple_model_first_200")
    parser.add_argument('--results_location', type=str, default="EntityDetection\\Results\\")
    parser.add_argument('--mode', type=str, default="LSTM")

    args = parser.parse_args()
    main(args)