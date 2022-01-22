# Text Preprocessing

First step is to download all the Budget Speeches and sort them in their appropriate folder

``` PDFConverter.py ```

- Used to convert PDF to text files (Converts both English and Maltese)

``` DocumentGenerator.py ```

- Used to convert text files from PDFConverter.py and generates the sentences that are part of the document
- Stores new sentences into json file

``` KGConstructors.py ```

- Used to annotate each sentence by separating it into a triple
- Stores entries into a json file separate from the documents


``` KGLoader.py ```

- Loads the data from the Wikidata and/or Budgets onto the respective knowledge graph

``` ExampleDataKGLoader.py ```

- The same a KGLoader.py but this has specific data for testing purposes

``` FreebaseRestructure.py ```

- Is there to transfer the .pk files of the graphs of the Freebase database to the Triple structure that we provide


``` FreebaseRestructureCSV.py ```

- Is there to transfer the .pk files of the graphs of the Freebase database to the CSV structure that we require to load them into NEO4J

To load into neo4j do the following in neo4j desktop:
Store files in the required folder

To create nodes:
``` LOAD CSV WITH HEADERS FROM "file:///FB2M_entities0.csv" AS line CREATE (n:Entity {ID: line.id, Text: line.text, Type: line.type}) ```

To create index:
``` CREATE INDEX id_index For (n:Entity) ON(n.ID) ```

To create relations between nodes:
``` 
LOAD CSV WITH HEADERS FROM "file:///FB2M_relations0.csv" AS line
MATCH (n:Entity {ID:line.subject_id})
MATCH (m:Entity {ID:line.object_id})
CALL apoc.create.relationship(n, line.relation, {}, m) YIELD rel
RETURN * 
```