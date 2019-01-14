import cv2
from PIL import Image
from itertools import islice


# processing of an image
def processing(path, thresh_1=235, thresh_2=240):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    ret, thresh1 = cv2.threshold(gray, thresh_1, 255, cv2.THRESH_BINARY_INV)
    dilate = cv2.dilate(thresh1, None, iterations=2)
    _, contours, _ = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contorList = []
    for contour in contours:
        contorList.append((contour, cv2.contourArea(contour)))
    sortedContours = sorted(contorList, key=lambda x: x[1], reverse=True)
    maxCNT = sortedContours[0][0]
    x, y, w, h = cv2.boundingRect(maxCNT)
    cropped = cv2.resize(image[y:y + h, x:x + w], (40, 40))
    thresh = thresh_2
    normalizedImage = cv2.threshold(cropped, thresh, 255, cv2.THRESH_BINARY)[1]
    pilImage = Image.fromarray(normalizedImage)
    a, b, c, d = path.split('/')
    finalImage = pilImage.convert('1', dither=Image.NONE)
    finalImage.save('../annotations/' + c + 'Processed/' + d)
    return finalImage


# processes a list of images in the annotations
def processImages(path, annotations, fileType):
    print 'Processing {} images...'.format(fileType)
    imagelist = []
    for line in islice(annotations, 1, None):
        line = line.split()
        imagelist.append(processing(path + fileType + '/' + line[0]))

    return imagelist
