import numpy as np
import cv2
import imutils
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border


def find_grid(image, debug=False):
    #apply adaptive thr. & invert
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7,7), 3)
    thresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2. THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    if debug:
        cv2.imshow("Grid thresh", thresh)
        cv2.waitKey(0)
    #FIND CONTOURS IN THRESHED
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key= cv2.contourArea, reverse=True)

    gridContour = None

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            gridContour = approx
            break

    if gridContour is None:
        raise Exception("Could not find grid outline.")

    if debug:
        out = image.copy()
        cv2.drawContours(out, [gridContour], -1, (0, 255, 0), 2)
        cv2.imshow("Sudoku Grid Outline", out)
        cv2.waitKey(0)

    #warping
    grid = four_point_transform(image, gridContour.reshape(4,2))
    warped = four_point_transform(imgGray, gridContour.reshape(4,2))

    if debug:
        cv2.imshow("Grid Transform", grid)
        cv2.waitKey(0)
    return (grid, warped)

def extract_digit(cell, debug=False):
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    if debug:
        cv2.imshow("Cell Thresh", thresh)
        cv2.waitKey(0)

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    #if contours = 0, then there is no digit in cell
    if len(contours) == 0:
        return None
    #otherwise find the biggest contour in cell and create a mask for it
    c = max(contours, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    #compute the percentage of masked pixels relative to the total area
    (h, w) = thresh.shape
    percent = cv2.countNonZero(mask) / float(w * h)

    #if percentage is less than 30%, it is noise so discard
    if percent < 0.03:
        return None
    #apply the mask to the thresholded cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    if debug:
        cv2.imshow("Digit", digit)
        cv2.waitKey(0)

    return digit

