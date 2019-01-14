from itertools import islice

from KNNClassifier import classifyByKNN
from imageProcessor import processImages
from naiveBayesClassifier import classifyByNaiveBayes
from rowColumnClassifier import classifyByRowColumn


def writeToFile(fileName, resultDigits, testFile):
    i = 0
    print 'Writing to file...'
    testFile.seek(0)
    with open('../results/{}-results.txt'.format(fileName), 'a+') as resultsFile:
        resultsFile.write('FileName\tPredicted Class\n')
        for line in islice(testFile, 1, None):
            line = line.split()
            resultsFile.write((line[0] + '\t' + str(resultDigits[i])) + '\n')
            i += 1


def combinedClassifiers(list1, list2, list3):
    print '\nRunning combined classifiers...'
    results = []
    for i in range(len(list1)):
        if list1[i] == list2[i]:
            results.append(list1[i])
        elif list2[i] == list3[i]:
            results.append(list2[i])
        else:
            results.append(list1[i])

    return results


def runEvalSetTests():
    path = '../annotations/'
    trainFile = open(path + 'train.txt', 'r')
    testFile = open(path + 'FileList-Eval.txt', 'r')

    trainImages = processImages(path, trainFile, 'train')
    testImages = processImages(path, testFile, 'Eval')

    resultDigits1 = classifyByNaiveBayes(trainFile, testImages)
    writeToFile('Naive-Bayes', resultDigits1, testFile)

    resultDigits2 = classifyByKNN(trainFile, trainImages, testImages)
    writeToFile('KNN', resultDigits2, testFile)

    resultDigits3 = classifyByRowColumn(trainFile, testImages)
    writeToFile('Row-Column', resultDigits3, testFile)

    resultDigits4 = combinedClassifiers(resultDigits1, resultDigits2, resultDigits3)
    writeToFile('Combined', resultDigits4, testFile)


"""
IMPORTANT: OpenCV is used for image processing. USE pip install opencv-python --user to install.
"""
if __name__ == '__main__':
    runEvalSetTests()
