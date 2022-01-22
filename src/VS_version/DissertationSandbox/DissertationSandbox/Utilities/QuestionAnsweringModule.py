from __future__ import unicode_literals, print_function

import sys

sys.path.insert(0, "Utilities")

import stanza
import plac
import nltk
import random
import spacy

from pathlib import Path
from nltk.chunk import conlltags2tree, tree2conlltags, ne_chunk
from nltk.tag.stanford import StanfordNERTagger
from nltk.tokenize import word_tokenize
from spacy.training.example import Example
from spacy.util import minibatch, compounding
from tqdm import tqdm

from Neo4jConnector import *


class StanzaEntityDetection:
    def __init__(self, language):
        stanza.download(language)
        self.nlp = stanza.Pipeline(lang=language, processors="tokenize,ner")
        print("Set up Stanza operator!")

    def detect(self, question):
        doc = nlp(question_text)
        print(*[f'entity: {ent.text}\ttype: {ent.type}' for ent in doc.ents], sep='\n')
        return doc.ents

class NLTKEntityDetection:
    def __init__(self):
        print("Set up NLTK operator!")

    def detect(self, question):
        ne_tree = ne_chunk(pos_tag(word_tokenize(question)))
        ne_in_sent = []
        for subtree in ne_tree:
            if type(subtree) == Tree:
                ne_label = subtree.label()
                ne_string = " ".join([token for token, pos in subtree.leaves()])
                ne_in_sent.append((ne_string, ne_label))
                
        return ne_in_sent

class SpacyEntityDetection:
    def __init__(self):
        self.model = None
        self.output_dir = "models\\EntityDetection"
        self.n_iter = 10
        print("Set up Spacy operator!")
        
    def createModel(self, training_data):
        if self.model is not None:
            self.nlp = spacy.load(model)
            print("Loaded model")
        else:
            self.nlp = spacy.blank('en')
            print("Created blank 'en' model")

        if 'ner' not in self.nlp.pipe_names:
            self.nlp.add_pipe('ner', last=True)
        else:
            self.ner = self.nlp.get_pipe('ner')

        # for _, annotations in training_data:
        #     for ent in annotations.get('entities'):
        #         self.ner.add_label(ent[2])

        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training()
            for itn in range(self.n_iter):
                section_data = training_data[:3200]
                random.shuffle(training_data)
                batches = minibatch(training_data, size=128)
                losses = {}
                for batch in tqdm(batches):
                    text, annotations = zip(*batch)
                    #doc = self.nlp.make_doc(text)
                    examples = [Example.from_dict(self.nlp.make_doc(t), a) for t,a in batch]
                    self.nlp.update(examples, drop=0.2, sgd=optimizer, losses=losses)
                print(losses)  

        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
            if not self.output_dir.exists():
                self.output_dir.mkdir()
            self.nlp.to_disk(self.output_dir)
            print("Saved model to ", self.output_dir)

class BruteForceEntityDetector:
    def __init__(self, database):
        self.database = database
        self.neo4jConnector = Neo4jConnector(uri="bolt://localhost:7687", user="admin", pwd="password")

    def detect(self, question):
        # tokenize the question
        tokens = word_tokenize(question)
        
