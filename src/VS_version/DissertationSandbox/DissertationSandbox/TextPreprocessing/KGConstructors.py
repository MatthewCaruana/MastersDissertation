import os
from openie import StanfordOpenIE
import glob
import json
import nltk
import time

def main():
    location = "..\..\..\..\Data\MaltaBudgets\English"
    
    os.chdir(location)
    
    openie_client = StanfordOpenIE()
    print(openie_client)
        
    for file in glob.glob("*.json"):
        f = open(file,'r', encoding= "utf-8")
        file_triples = []
        sentence_list = json.load(f)
        for sentence in sentence_list:
            tokens = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(tokens)
            entities = nltk.ne_chunk(tagged)
    
            triples = []
    
            if len(entities) > 0:
                for triple in openie_client.annotate(sentence):
                    if len(triple) > 0:
                        triples.append(triple)
    
            file_triples.append(triples)
        
        json_file = open("triples/" + file.replace(".json", "-triples.json"), 'w', encoding="utf-8")
        json_file.write(json.dumps(file_triples))
        json_file.close()

if __name__ == "__main__":
    main()