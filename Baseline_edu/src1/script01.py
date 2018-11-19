import _pickle as pickle
from CorpusClass import Corpus
    
myPath = "../dataset/"
corpus = Corpus(myPath)
with open(myPath + "corpus.pickle", "wb") as f:
    pickle.dump(corpus, f)













