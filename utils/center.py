import time
import numpy as np
import cv2
from skimage import io
from skimage.util import pad
from skimage.draw import circle


def mask2contour(img):
    """
    Input:  Single channel uint8 image
    Ouput:  Contour list, non-approximated
    Step:   Unconnect adjacent cell masks then use cv2.findContours
    """
    img_dilate = cv2.dilate(img, np.ones((3,3),np.uint8), iterations=1)
    img_unconnected = np.where(img==0, 0, 255).astype('uint8')
    img_unconnected = np.where(img_dilate-img != 0, 0, img_unconnected).astype('uint8')

    ret = cv2.findContours(img_unconnected, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = ret[-2]
    print(len(contours), 'contours found!')
    return contours


def ROI(image, contourlist, padded=True):
    """
    Crop ROI with same label
    Return list ROIs and list of BBOXes (minX, maxX, minY, maxY)
    """
    roilist = []
    BBOXlist = []

    for ct in contourlist:
        bbox = cv2.boundingRect(ct)
        bbox = (bbox[1],bbox[0],bbox[3],bbox[2])
        coordiBBOX = (bbox[0], bbox[0]+bbox[2], bbox[1], bbox[1]+bbox[3])
        roi = image[bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3]]

        (values,counts) = np.unique(roi,return_counts=True)
        label = values[np.argmax(counts)]
        roi = np.where(roi == label, roi, 0)
        if padded:
            roi = pad(roi, pad_width=1, mode="constant")
        roilist.append(roi)
        BBOXlist.append(coordiBBOX)
    return roilist, BBOXlist


def checkConti(arrayC):
    """
    Check if the labelled pixels are continuous
    along in the vector.
    00111111100 -> True
    00111001100 -> False
    """
    diff = arrayC - np.roll(arrayC, 1)
    changeFlag = np.count_nonzero(diff)
    if changeFlag > 2:
        return False
    else:
        return True


def roi2origin(x, y, coordiBBOX, padded=True):
    """
    Coordinates transformation
    from ROI to original large image
    """
    if padded:
        x_origin = x + coordiBBOX[0] - 1
        y_origin = y + coordiBBOX[2] - 1
    else:
        x_origin = x + coordiBBOX[0]
        y_origin = y + coordiBBOX[2]
    return x_origin, y_origin



def main(filename): 
    t = time.time()
    img = io.imread(filename)

    ROIlist, BBOXlist = ROI(img, mask2contour(img))
    for roi, coordiBBOX in zip(ROIlist, BBOXlist):
        contiX = []
        contiY = []
        midX_origin = []
        midY_origin = []

        for x in range(roi.shape[0]):
            if checkConti(roi[x,:]):
                contiX.append(x)
        for y in range(roi.shape[1]):
            if checkConti(roi[:,y].T):
                contiY.append(y)
        midX = contiX[ np.int(len(contiX)/2) ]
        midY = contiY[ np.int(len(contiY)/2) ]    

        x_origin, y_origin = roi2origin(midX, midY, coordiBBOX, padded=True)
        midX_origin.append(x_origin)
        midY_origin.append(y_origin)

        for x, y in zip(midX_origin, midY_origin):
            rr, cc = circle(x, y, 3)
            img[rr, cc] = 0

    print(time.time() - t)
    io.imshow(img)
    io.show()

main(filename="img1.png")