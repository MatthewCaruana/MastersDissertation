
from Utilities.Triples import *

class Transformer:
    @staticmethod
    def QueryResponseToTriple(response_data):
        triple_list = []
        for response in response_data:
            triple = Triple(response[0], response[1], response[2])