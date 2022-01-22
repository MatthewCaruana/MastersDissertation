from argparse import ArgumentParser

import nltk
from nltk.tag.stanford import StanfordNERTagger

def load_questions(data_dir):
    file = open(data_dir + "all.txt", 'r', encoding="utf-8")
    records = []
    for file_line in file:
        records.append(file_line[:-1].split("\t"))
    file.close()
    return records

def main(args):
    questions = load_questions(args.data_dir)


if __name__ == "__main__":
    parser = ArgumentParser(description="Named Entity Recognition")
    parser.add_argument('--type', type=str, default="SimpleQuestions", help="SimpleQuestions/")
    parser.add_argument('--data_dir', type=str, default="data/QuestionAnswering/processed_simplequestions_dataset/")
    parser.add_argument('--mode', type=str, default="test")

    args = parser.parse_args()

    main(args)