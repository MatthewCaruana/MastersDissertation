# Entity Linking

The process of linking the entity to an actual node within the knowledge graph (Not Neural Network)

## Mohammed's Implementation

Uses fuzzy string matching by using an inverted index over n-grams in an entity's name. They start with a n = 3 factor until an exact match is found, going lower if nothing is found. Once all candidate entities have been gathered, they are then ranked by Levenshtein Distance to the MID's canonical label.

## My implementation

PageRank?