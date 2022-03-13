import sys

sys.path.insert(0, "Utilities")

import argparse
import os

from Triples import *

def main(args):
    # load each file and place all relations into 
    files = [f for f in os.listdir(args.triples_loc)]

    for file in files:
        json_file = open(args.triples_loc + file, 'r')
        json_content = json.loads(json_file.read())

        year = file[1:5]

        properties = [Property("Year", year)]

        for triple in json_content:
            if triple:
                triple_neo4j = Triple(str(triple[0]["subject"]), triple[0]["relation"], triple[0]["object"])
                connector.AddTriple(triple_neo4j, properties)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transforming the Malta Budgets data to our Triple structure")
    parser.add_argument('--triples_loc',type=str, default="data\\MaltaBudgets\\English\\triples\\")
    parser.add_argument('--output', type=str, default="data\\MaltaBudgets\\English\\triples\\")
    parser.add_argument('--type', type=str, default="MaltaBudgets")

    args = parser.parse_args()

    main(args)