# In this section the processing and evaluation of the Relation Prediction module will be tested
import argparse
import sys
import pandas as pd
import tqdm

sys.path.insert(0, "Utilities")

from RelationPredictionModule import *
from DatasetUtils import *


def relation_prediction(data, model, mode):
    print("Starting Relation Prediction")
    results = []
    results_60p = []
    results_70p = []
    results_80p = []
    results_90p = []

    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        row_result = model.detect(row[3])


def evaluate(actual, predicted, mode):
    print("Starting Evaluation for " + mode)

    predicted = len(predicted)
    total = len(actual)
    correct = 0

    for index, predicted_row in predicted.iterrows():
        actual_row = actual.loc[actual[0] == predicted[0]] # Search by Name

        if actual_row[2] == predicted_row[1]:
            correct += 1

    if predicted == 0:
        precision = 0
    else:
        precision = correct / predicted

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

    # Load files from dataset
    valid = pd.read_csv(args.location + "valid.txt", sep="\t", header=None)
    valid.colums = ["Name", "EntityID", "EntityText", "Relation", "ResultID", "Question", "Entities"]
    print(valid)
    valid = DatasetUtils.FormatSimpleQuestionsForRelationPrediction(valid)

    test = pd.read_csv(args.location + "test.txt", sep="\t", header=None)
    test.colums = ["Name", "EntityID", "EntityText", "Relation", "ResultID", "Question", "Entities"]
    test = DatasetUtils.FormatSimpleQuestionsForRelationPrediction(test)

    print(valid)
    print(test)

    model = BruteForceRelationPrediction()

    valid_results = relation_prediction(valid, model, "valid")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entity Detection module training/testing framework")
    parser.add_argument('--location', type=str, default='data\\QuestionAnswering\\processed_simplequestions_dataset\\')
    parser.add_argument('--mode', type=str, default="STANZA", help="STANZA/NLTK/BF")
    parser.add_argument('--database', type=str, default="fb2m")
    parser.add_argument('--extraction',type=str, default="STANZA", help="STANZA/NLTK, STANZA is the required one")
    parser.add_argument('--language',type=str, default="en")

    args = parser.parse_args()
    main(args)