import numpy as np
from itertools import islice


# Needs to be more optimized but no time!
def calculateHammingDistance(trainFile, testImages, trainImages):
    trainDictionary = {}
    hammingDistance = []
    results = []

    for image in testImages:
        hammingDistance[:] = []
        trainDictionary.clear()
        trainFile.seek(0)
        i = 0
        for line in islice(trainFile, 1, None):
            line = line.split()
            fileName = line[0]
            fileDigit = int(line[1])
            trainDictionary[fileName] = fileDigit
            hammingDistance.append((fileName, np.logical_xor(np.asarray(image), np.asarray(trainImages[i])).sum()))
            i += 1

        sortedHamming = sorted(hammingDistance, key=lambda x: x[1])

        K = 3
        possibleDigits = []
        for i in range(K):
            possibleDigits.append((sortedHamming[i][1], trainDictionary[sortedHamming[i][0]]))

        # Simple Break of ties
        if possibleDigits[0][1] == possibleDigits[1][1]:
            results.append(possibleDigits[0][1])
        elif possibleDigits[1][1] == possibleDigits[2][1]:
            results.append(possibleDigits[1][1])
        else:
            results.append(possibleDigits[0][1])

        possibleDigits[:] = []
    return results


def classifyByKNN(trainFile, trainImages, testImages):
    print '\nRunning KNN Classifier'
    trainFile.seek(0)

    print 'Calculating hamming distances (takes around 2 mins)'
    resultDigits = calculateHammingDistance(trainFile, testImages, trainImages)

    return resultDigits
