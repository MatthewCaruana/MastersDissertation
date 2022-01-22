import argparse
from argparse import ArgumentParser

def main(args):
    entities = []
    question_entities = []
    if args.file_type == "Freebase":
        file = open(args.file_location + args.file_name, 'r', encoding="utf-8")
        for line in file:
            entities.append(line[:-1].split("\t"))

        file.close()

        entities_file = open(args.file_location + args.file_specific_type + ".tsv", "w", encoding="utf-8")

        for entity in entities:
            entities_file.write(entity[2] + "\tI\n")

        entities_file.close()
    elif args.file_type == "SimpleQuestions":
        file = open(args.file_location + args.file_name, 'r', encoding="utf-8")
        for line in file:
            question_entities.append(line[:-1].split("\t"))

        file.close()

        entities = []
        for question_all in question_entities:
            question = question_all[5]
            entity = question_all[2]
            entity_start_index = question.find(entity)
            entity_end_index = len(entity) + entity_start_index
            entities.append((question, {'entities': [(entity_start_index, entity_end_index, 'Entity')]}))

        question_file = open(args.file_location + args.file_specific_type, 'w', encoding="utf-8")
        for entity_input in entities:
            question_file.write(entity_input + "\n")

        question_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract the Entities into a readable TSV file for model creation.")
    parser.add_argument('--file_location', type=str, default='data\\QuestionAnswering\\processed_simplequestions_dataset\\', help="location of file that needs to be transformed")
    parser.add_argument('--file_name', type=str, default='valid.txt')
    parser.add_argument('--file_specific_type', type=str, default="valid")
    parser.add_argument('--file_type', type=str, default="SimpleQuestions")

    args = parser.parse_args()
    main(args)