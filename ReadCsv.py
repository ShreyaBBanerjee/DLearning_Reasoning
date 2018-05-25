import pandas as pd
import pickle
import autograd.numpy as np
from autograd import grad

train = pd.read_csv("./train.csv",header=None, names= ["id","c1","c2","c3", "c4"])
test = pd.read_csv("./test.csv",header=None, names= ["id","c1","c2","c3", "c4"])
samples = 50000
testsamples = 50000
def form_image():
    trainformula = np.zeros((samples,4,4))
    trainlabel = np.zeros(samples)
    testformula = np.zeros((testsamples,4,4))
    testlabel = np.zeros(testsamples)

    for id in range(testsamples):
        trainproblemChunk = train[train["id"] == ("P" + str(id))][["c1", "c2", "c3", "c4"]]
        testproblemChunk = test[test["id"] == ("P" + str(id))][["c1", "c2", "c3", "c4"]]
        trainformula[id] = trainproblemChunk[0:len(trainproblemChunk) - 1].values.astype(float)
        trainlabel[id] = trainproblemChunk["c4"].values[4]
        testformula[id] = testproblemChunk[0:len(testproblemChunk) - 1].values.astype(float)
        testlabel[id] = testproblemChunk["c4"].values[4]

    filehandler = open("train_formula.txt", "wb")
    pickle.dump(trainformula, filehandler)
    filehandler.close()
    filehandler = open("train_label.txt", "wb")
    pickle.dump(trainlabel, filehandler)
    filehandler.close()
    filehandler = open("test_formula.txt", "wb")
    pickle.dump(testformula, filehandler)
    filehandler.close()
    filehandler = open("test_label.txt", "wb")
    pickle.dump(testlabel, filehandler)
    filehandler.close()

form_image()