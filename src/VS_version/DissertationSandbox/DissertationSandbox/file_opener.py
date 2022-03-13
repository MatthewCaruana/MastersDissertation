import pickle
import json
import pandas as pd
import nltk
import sys

from nltk.tokenize import word_tokenize

file = open("data//QuestionAnswering//sq_glove300d.pt", "r", encoding="utf-8")
pickle.load(file)
#count = 0

for x in file:
    #print(pickle.load(x)
    print(x)

#    count = count + 1