import argparse

sys.path.insert(0, "Utilities")

from Neo4jConnector import *

def main(args):
    neo4j_connector = Neo4jConnector(uri=args.neo4j_loc, user=args.neo4j_username, pwd=args.neo4j_password)

    response_content = neoConnector.query("MATCH (n)-[r]->(m) WHERE m.name CONTAINS '100' RETURN n,r,m LIMIT 25")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Database Generation for Maltese Budgets/Press Releases")
    parser.add_argument('--location', type=str, default='data\\MaltaBudgets\\processed_simplequestions_dataset\\')
    parser.add_argument('--language',type=str, default="English")
    parser.add_argument('--dataset', type=str, default="SimpleQuestions")

    parser.add_argument('--neo4j_loc', type=str, default="bolt://localhost:7687")
    parser.add_argument('--neo4j_username', type=str, default="admin")
    parser.add_argument('--neo4j_password', type=str, default="password")

    args = parser.parse_args()
    main(args)