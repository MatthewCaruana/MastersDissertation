import argparse
import sys
import json
import csv

def main_pos(args):
    doi_sentence_file = open(args.data_location, 'r', encoding="utf-8")
    doi_sentences = json.load(doi_sentence_file)
    doi_sentence_file.close()

    tokenized_sentences = []
    
    doi_tokenized_file = open(args.result_location + "tokens-2019.tsv", "w", encoding="utf-8", newline="")
    writer= csv.writer(doi_tokenized_file, delimiter="\t")
    for doi_sentence in doi_sentences:
        #writer.writerow(["#" + doi_sentence])
        doi_sentence = doi_sentence.replace("-", "- ")
        #tokenize
        tokens = doi_sentence.split()
        count = 1
        if len(tokens) > 1:
            for token in tokens:
                writer.writerow([count, token, "DET", "_"])
                count = count + 1

            if not tokens == []:
                writer.writerow([])

    doi_tokenized_file.close()

def main_dep(args):


    doi_upos_file = open(args.upos_data_location, 'r', encoding="utf-8")
    reader = csv.reader(doi_upos_file, delimiter="\t")
    doi_pos_list = []
    for row in reader:
        if row == []:
            doi_pos_list.append([])
        else:
            doi_pos_list.append([row[0], row[1], "_", row[2], "-", "_", 0, "punct", "_", "_"])

    doi_upos_file.close()

    doi_xpos_file = open(args.xpos_data_location, 'r', encoding="utf-8")
    reader = csv.reader(doi_xpos_file, delimiter="\t")
    i = 0
    for row in reader:
        if row == []:
            i = i + 1
        elif doi_pos_list[i][0] == row[0] and doi_pos_list[i][1] == row[1]:
            doi_pos_list[i][4] = row[2]
            i = i + 1

    doi_xpos_file.close()

    doi_dep_file = open(args.result_location + "pos-2019-dep.tsv", "w", encoding="utf-8", newline="")
    writer = csv.writer(doi_dep_file, delimiter="\t")
    for doi_pos in doi_pos_list:
        writer.writerow(doi_pos)

    doi_dep_file.close()

def main_ner(args):
    doi_sentence_file = open(args.data_location, 'r', encoding="utf-8")
    doi_sentences = json.load(doi_sentence_file)
    doi_sentence_file.close()

    tokenized_sentences = []
    
    doi_tokenized_file = open(args.result_location + "tokens-2019-for-ner.tsv", "w", encoding="utf-8", newline="")
    writer= csv.writer(doi_tokenized_file, delimiter="\t")
    for doi_sentence in doi_sentences:
        #writer.writerow(["#" + doi_sentence])
        doi_sentence = doi_sentence.replace("-", "- ")
        #tokenize
        tokens = doi_sentence.split()
        if len(tokens) > 1:
            for token in tokens:
                writer.writerow(["mt:"+ token, "O"])

            if not tokens == []:
                writer.writerow([])

    doi_tokenized_file.close()
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraction of Maltese Triples from our datasources which are DOI and Budget Data")
    parser.add_argument('--data_location', type=str, default='data\\DOI\\triples\\mt\\sentences-2019.txt')
    parser.add_argument('--upos_data_location', type=str, default='data\\DOI\\triples\\mt\\upos-2019.tsv')
    parser.add_argument('--xpos_data_location', type=str, default='data\\DOI\\triples\\mt\\xpos-2019.tsv')
    parser.add_argument('--result_location', type=str, default='data\\DOI\\triples\\mt\\')
    parser.add_argument('--stage', type=str, default="NER") #POS, DEP, NER

    args = parser.parse_args()

    if(args.stage == "POS"):
        main_pos(args)
    elif(args.stage == "DEP"):
        main_dep(args)
    elif(args.stage == "NER"):
        main_ner(args)
