# In this section the processing and evaluation of the Entity Detection module will be tested
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

def save_checkpoint(dataframe, file_name):
    temp = pd.DataFrame(dataframe, columns=["Name", "EntityText", "Score"])
    temp.to_csv(file_name)

def train_model(data, database_name):
    print("Starting Training the Entity Detection Model for " + database_name)


def entity_detection(data, model, bf_model, mode):
    print("Starting Entity Detection for " + mode)
    results = []
    results_60p = []
    results_70p = []
    results_80p = []
    results_90p = []

    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        row_response = model.detect(row[5])

        if len(row_response) == 0:
            entity = bf_model.detect(row[5], 3)
        else:
            # locate node in database
            entity = bf_model.detect(row_response[0].text, 3)

        if entity:
            score = entity[2]
            if score >= 0.9:
                results_90p.append([row[0],entity[1][0].get("Text"), entity[2]])
            elif score >= 0.8:
                results_80p.append([row[0],entity[1][0].get("Text"), entity[2]])
            elif score >= 0.7:
                results_70p.append([row[0],entity[1][0].get("Text"), entity[2]])
            elif score >= 0.6:
                results_60p.append([row[0],entity[1][0].get("Text"), entity[2]])
            
            results.append([row[0],entity[1][0].get("Text"), entity[2]])

        if index % 1000 == 0 and index > 0:
            if len(results) > 0:
                save_checkpoint(results, args.location + "checkpoints\\ed_" + mode + "_responses_" + str(index) + "_all.csv")
                save_checkpoint(results_60p, args.location + "checkpoints\\ed_" + mode + "_responses_" + str(index) + "_60p.csv")
                save_checkpoint(results_70p, args.location + "checkpoints\\ed_" + mode + "_responses_" + str(index) + "_70p.csv")
                save_checkpoint(results_80p, args.location + "checkpoints\\ed_" + mode + "_responses_" + str(index) + "_80p.csv")
                save_checkpoint(results_90p, args.location + "checkpoints\\ed_" + mode + "_responses_" + str(index) + "_90p.csv")

    save_checkpoint(results, args.location + "ed_" + mode + "_responses_final_all.csv")
    save_checkpoint(results_60p, args.location + "ed_" + mode + "_responses_final_60p.csv")
    save_checkpoint(results_70p, args.location + "ed_" + mode + "_responses_final_70p.csv")
    save_checkpoint(results_80p, args.location + "ed_" + mode + "_responses_final_80p.csv")
    save_checkpoint(results_90p, args.location + "ed_" + mode + "_responses_final_90p.csv")

    return pd.DataFrame(results, columns=["Name", "EntityText", "Score"])

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

def main(args):
    print(args)

    if args.do_training == True:
        train = pd.read_csv(args.location + "train.txt", sep="\t", header= None)
        train.colums = ["Name", "EntityID", "EntityText", "Relation", "ResultID", "Question", "Entities"]
        train = DatasetUtils.FormatSimpleQuestionsForEntityDetection(valid)
        train_model(train)

    # Load files from dataset
    valid = pd.read_csv(args.location + "valid.txt", sep="\t", header=None)
    valid.colums = ["Name", "EntityID", "EntityText", "Relation", "ResultID", "Question", "Entities"]
    valid = DatasetUtils.FormatSimpleQuestionsForEntityDetection(valid)

    test = pd.read_csv(args.location + "test.txt", sep="\t", header=None)
    test.colums = ["Name", "EntityID", "EntityText", "Relation", "ResultID", "Question", "Entities"]
    test = DatasetUtils.FormatSimpleQuestionsForEntityDetection(test)

    # Open Entity Detector method type
    if args.mode == "STANZA":
        model = StanzaEntityDetection(args.language)
    elif args.mode == "NLTK":
        model = NLTKEntityDetection()
    elif args.mode == "Neural":
        model = NeuralEntityDetection()

    bf_model = BruteForceEntityDetector(args.database)

    if args.load_valid:
        results_valid = load_file(args.location + "ed_valid_responses_final_all.csv")
    else:
        results_valid = entity_detection(valid, model, bf_model, "valid")

    if args.load_test:
        results_test = load_file(args.location + "ed_test_responses_final_all.csv")
    else:
        results_test = entity_detection(test, model, bf_model, "test")

    # evaluate predicted with actual
    valid_precision, valid_recall, valid_f1 = evaluate(valid, results_valid, "valid")
    test_precision, test_recall, test_f1 = evaluate(test, results_test, "test")

    # write to files
    valid_file = open(args.location + "ed_valid_results.txt", "w")
    valid_file.write("Valid")
    valid_file.write("Precision:\t" + str(valid_precision))
    valid_file.write("Recall:\t" + str(valid_recall))
    valid_file.write("F1-score:\t" + str(valid_f1))
    valid_file.close()

    test_file = open(args.location + "ed_test_results.txt", "w")
    test_file.write("Test")
    test_file.write("Precision:\t" + str(test_precision))
    test_file.write("Recall:\t" + str(test_recall))
    test_file.write("F1-score:\t" + str(test_f1))
    test_file.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entity Detection module training/testing framework")
    parser.add_argument('--location', type=str, default='data\\QuestionAnswering\\processed_simplequestions_dataset\\')
    parser.add_argument('--mode', type=str, default="Neural", help="STANZA/NLTK/Neural/BF")
    parser.add_argument('--database', type=str, default="fb2m")
    parser.add_argument('--extraction',type=str, default="STANZA", help="STANZA/NLTK, STANZA is the required one")
    parser.add_argument('--language',type=str, default="en")
    parser.add_argument('--load_valid',type=bool, default=True)
    parser.add_argument('--load_test',type=bool, default=True)
    parser.add_argument('--do_training', type=bool, default=True)

    args = parser.parse_args()
    main(args)