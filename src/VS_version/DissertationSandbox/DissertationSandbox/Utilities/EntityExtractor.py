import nltk
import stanza
from openie import StanfordOpenIE
import zeep
from nltk.tree import Tree

class StanzaEntityExtractor:
    def __init__(self, language):
        stanza.download(language)
        self.nlp = stanza.Pipeline(lang=language, processors="tokenize,ner")
        print("Stanza Entity Extractor Initialised!")

    def extract_entities(self, sentence):
        doc = self.nlp(sentence)
        print(*[f'entity: {ent.text}\ttype: {ent.type}' for ent in doc.ents], sep='\n')
        return doc.ents

class NLTKEntityExtractor:
    def __init__(self):
        print("NLTK Entity Extractor Initialised!")
        
    def extract_entities(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        entities = nltk.ne_chunk(tagged)

        return entities

class SVOEntityExtractor:
    def __init__(self, language):
        self.language = language
        print("SVO Entity Extractor Initialised!")
        self.maltese_client = "http://metanet4u.research.um.edu.mt/services/MtPOS?wsdl"

        if language == "mt":
            self.client = zeep.Client(wsdl = self.maltese_client)

        self.patterns = """P: {<DD>?<MJ.*>*<NP.*>}"""

    def extract_entities(self, sentence):
        # parse sentence
        if self.language == "mt":
            parsed_sentence = self.client.service.tagParagraphReturn(sentence)
            tokens = parsed_sentence.split(" ")[:-1]
            tagged = [tuple(token.split("_")) for token in tokens]

            chunker = nltk.RegexpParser(self.patterns)

            chunks = chunker.parse(tagged)
        elif self.language == "en":
            tokens = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(tokens)

            chunks = nltk.ne_chunk(tagged)

        continuous_chunk = []
        current_chunk = []

        for i in chunks:
            if type(i) == Tree:
                current_chunk.append(" ".join([token for token, pos in i.leaves()]))
            elif current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue

        if continuous_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)

        return continuous_chunk

        print(tagged)
        # SVO extraction     
        return chunks



if __name__ == "__main__":
    svo_entity = SVOEntityExtractor("mt")

    svo_entity.extract_entities("Skema ġdida minn Ambjent Malta (AM) ser tkun qed tħeġġeġ studenti tas-Snin 4, 5 u 6 fi skejjel tal-gvern, privati u tal-knisja biex flimkien iħawlu siġra fil-21 ta' Marzu biex jiġi ċċelebrat il-bidu tar-rebbiegħa.")
    #svo_entity.extract_entities("A new scheme by Ambjent Malta (AM) will encourage Year 4, 5 and 6 students in state, private and church schools to collectively sow a tree on the 21st of March to celebrate the beginning of spring. ")
