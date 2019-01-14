from multiprocessing import Pool

import numpy as np
from PIL import Image
from itertools import islice


def columnExtractor(fileList):
    imagelist = []
    for file in fileList:
        img = Image.open('../annotations/trainProcessed/' + file)
        imagelist.append(img)
    averagePixels = (~np.asarray(imagelist[0])).astype(int)
    for i in range(1, len(imagelist)):
        averagePixels += (~np.asarray(imagelist[i])).astype(int)
    averagePixels = averagePixels / float(len(imagelist))
    return np.sum(averagePixels, axis=0)


def rowExtractor(fileList):
    imagelist = []
    for file in fileList:
        img = Image.open('../annotations/trainProcessed/' + file)
        imagelist.append(img)
    averagePixels = (~np.asarray(imagelist[0])).astype(int)
    for i in range(1, len(imagelist)):
        averagePixels += (~np.asarray(imagelist[i])).astype(int)
    averagePixels = averagePixels / float(len(imagelist))
    return np.sum(averagePixels, axis=1)


def classifyByRowColumn(trainFile, testImages):
    print '\nRunning Row Column Classifier...'
    # These are the features of digits from 0 to 9
    print 'Extracting features...'

    fileLists = [[], [], [], [], [], [], [], [], [], []]
    trainFile.seek(0)
    for line in islice(trainFile, 1, None):
        line = line.split()
        fileLists[int(line[1])].append(line[0])

    # Number of processes
    p1 = Pool(processes=4)
    # These are the features of digits from 0 to 9
    cfeatures = p1.map(columnExtractor, fileLists)
    p2 = Pool(processes=4)
    rfeatures = p2.map(rowExtractor, fileLists)

    resultDigits = []
    print 'Calculating differences...'
    for image in testImages:
        columnFeatures = [[] for _ in range(40)]
        rowFeatures = [[] for _ in range(40)]

        ImageColumns = np.sum(~(np.asarray(image)), axis=0)
        ImageRows = np.sum(~(np.asarray(image)), axis=1)

        for i in range(40):
            for j in range(10):
                columnFeatures[i].append(abs(ImageColumns[i] - cfeatures[j][i]))
                rowFeatures[i].append(abs(ImageRows[i] - rfeatures[j][i]))

        combinedFeatures = 0.8 * np.sum(np.asarray(columnFeatures), axis=0) + np.sum(np.asarray(rowFeatures), axis=0)
        resultDigits.append(np.argmin(combinedFeatures))

    return resultDigits
