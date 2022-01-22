import pickle

file = open("../../../../Data/data/SimpleQuestions_v2/freebase-subsets/freebase-FB2M-graph.pk", "rb")
# file = open("../../../../Data/data/SimpleQuestions_v2/freebase-subsets/freebase-FB5M.txt")

print(pickle.load(file))

#for x in file:
    #print(pickle.load(x)
    #print(x)