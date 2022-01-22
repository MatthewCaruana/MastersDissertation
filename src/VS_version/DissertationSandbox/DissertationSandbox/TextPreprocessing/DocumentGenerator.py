# This file is meant to gather all text files and generate a json file that is meant to reformat the transformed PDFs into a more representable format
# We are recognising a sentence based on a set of criteria:
# 1. There is more than 1 word or character
# 2. If there is only 1 word it must have a full-stop

import os
import glob
import fitz
import json

def main():
    os.chdir('..\..\..\..\Data\MaltaBudgets\English')
    
    
    for file in glob.glob("*.txt"):
        sentences = []
        current_sentence = []
        f = open(file, encoding="utf-8")
        for line in f:
            words = line.split(" ")
            if len(words) > 1:
                for word in words:
                    if word != "\n" and word != "":
                        if '.' == word[-1]:
                            current_sentence.append(word)
                            sentences.append(''.join(current_sentence))
                            current_sentence = []
                        else:
                            current_sentence.append(word + " ")
            else:
                if '.' == words[-1]:
                    current_sentence.append(words)
                    sentences.append(''.join(current_sentence))
                    current_sentence = []
        f.close()
    
        json_file = file.replace("txt", "json")
    
        if os.path.isfile(json_file):
            os.remove(json_file)
    
        f = open(json_file, 'a')
        f.write(json.dumps(sentences))
        f.close()
    
if __name__ == "__main__":
    main()