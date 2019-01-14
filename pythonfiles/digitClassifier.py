from itertools import islice

from KNNClassifier import classifyByKNN
from imageProcessor import processImages
from naiveBayesClassifier import classifyByNaiveBayes
from rowColumnClassifier import classifyByRowColumn


def percentage(part, whole):
    return str(100 * float(part) / float(whole)) + '%'


def testClassifier(resultDigits, expectedDigits, classifier):
    testSize = len(resultDigits)
    correctDigits = 0
    wrongResults = []

    for i in range(testSize):
        if resultDigits[i] == expectedDigits[i]:
            correctDigits += 1
        else:
            wrongResults.append((expectedDigits[i], resultDigits[i]))

    print '\nThe ' + classifier + ' classifier resulted in ' + str(correctDigits) + ' correct digits out of ' + str(
        testSize) + ' i.e., accuracy of ' + percentage(correctDigits, testSize) + '.'

    choice = int(raw_input('Would you like to print the wrong results? (1/0): '))
    if choice:
        print 'The wrong results in format of (actual digit, detected digit)'
        for i in range(len(wrongResults)):
            if i % 15 == 0:
                print ''
            print str(wrongResults[i]) + '\t',
        print ''


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


# this will run all tests on the dev set
def runDevSetTests():
    path = '../annotations/'
    devFile = open(path + 'dev.txt', 'r')
    trainFile = open(path + 'train.txt', 'r')

    devImages = processImages(path, devFile, 'dev')
    trainImages = processImages(path, trainFile, 'train')

    devFile.seek(0)
    expectedDigits = []
    for line in islice(devFile, 1, None):
        line = line.split()
        if len(line) > 1:
            expectedDigits.append(int(line[1]))

    resultDigits1 = classifyByNaiveBayes(trainFile, devImages)
    testClassifier(resultDigits1, expectedDigits, 'Naive Bayes')

    resultDigits2 = classifyByKNN(trainFile, trainImages, devImages)
    testClassifier(resultDigits2, expectedDigits, 'K-Nearest Neighbour')

    resultDigits3 = classifyByRowColumn(trainFile, devImages)
    testClassifier(resultDigits3, expectedDigits, 'Row Column')

    resultDigits4 = combinedClassifiers(resultDigits1, resultDigits2, resultDigits3)
    testClassifier(resultDigits4, expectedDigits, 'Combined')


"""
IMPORTANT: OpenCV is used for image processing. USE pip install opencv-python --user to install.
"""
if __name__ == '__main__':
    runDevSetTests()



