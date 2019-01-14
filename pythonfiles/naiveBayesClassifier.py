from multiprocessing import Pool

import math
import numpy as np
from PIL import Image
from itertools import islice


def featureExtractor(fileList):
    imagelist = []
    for file in fileList:
        img = Image.open('../annotations/trainProcessed/' + file)
        imagelist.append(img)

    smoothingThresh = 1.0
    domain = 2.0
    sumedPixels = (~np.asarray(imagelist[0])).astype(int)

    for i in range(1, len(imagelist)):
        sumedPixels += (~np.asarray(imagelist[i])).astype(int)

    feature = (sumedPixels + smoothingThresh) / (len(imagelist) + domain * smoothingThresh)

    return feature.flatten()


def naivebayes(probabilites, imagelist):
    resultDigits = []
    for image in imagelist:
        imagedata = ~np.asarray(image).flatten()
        results = []
        for i in range(len(probabilites)):
            probability = 0
            for j in range(len(imagedata)):
                if imagedata[j]:
                    probability += math.log(probabilites[i][j])
                else:
                    probability += math.log(1 - probabilites[i][j])

            results.append((i, probability))

        sortedProb = sorted(results, key=lambda x: x[1], reverse=True)
        resultDigits.append(sortedProb[0][0])

    return resultDigits


def classifyByNaiveBayes(trainFile, testImages):
    print '\nRunning Naive Bayes classifier..'
    trainFile.seek(0)

    fileLists = [[] for _ in range(10)]
    for line in islice(trainFile, 1, None):
        line = line.split()
        fileLists[int(line[1])].append(line[0])

    print 'Extracting features...'
    # Number of processes
    p = Pool(processes=4)
    # These are the features of digits from 0 to 9
    probabilities = p.map(featureExtractor, fileLists)

    print 'Running classification...'
    resultDigits = naivebayes(probabilities, testImages)

    return resultDigits
