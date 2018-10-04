import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

# Process the image to get the edges
def edges( image ):
    image_rsz=cv2.resize(image, (800,800))
    image_blur=cv2.bilateralFilter(image_rsz, 9, sigmaSpace=0, sigmaColor=180)
    a=0.66*np.mean(image_blur)
    b=2*a
    #apply canny edge detector
    image_edges=cv2.Canny(image_blur,  a, b)
    return image_edges

# draw straight lines on the image based on Probabilistic Hough Line Transform
def draw_lines(image1, image2):
    lines =cv2.HoughLinesP(image1, 0.5, np.pi/360, 50, maxLineGap=100, minLineLength=10)
    #lines=emerge(lines)
    print (lines)
    if lines is not None:
        for element in lines:
            line= element[0]
            image_lines=cv2.line(image2, (line[0], line[1]), (line[2], line[3]), (0,255,0), 1)
    return image_lines

# draw straight lines on the image based on Standard hough Line Transform
def draw(image1, image2):
    linesS = cv2.HoughLines(image1, 0.5, np.pi / 360, 130)
    if linesS is not None:
        for i in range(0, len(linesS)):
            rho = linesS[i][0][0]
            theta = linesS[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            image_linesS=cv2.line(image2, pt1, pt2, (0,0,255), 1)
    return image_linesS
# a function to emerge the converging lines
def emerge(lines):
    if lines is not None:
        for i in range(0, len(lines)):
            line =lines[i][0]
            for j in range(i, len(lines)):
                if lines[j][0][0]+5>line[0]>lines[j][0][0]-5 and lines[j][0][1]+5>line[1]>lines[j][0][1]-5 and (lines[j][0][2]+5>line[2]>lines[j][0][2]-5 or lines[j][0][3]+5>line[3]>lines[j][0][3]-5):
                    np.delete(lines, lines[i])
    return lines
def main():
    original_image=cv2.imread('chessboard2.jpg')
    edged_image=edges(original_image)
    image_bgr=cv2.cvtColor(edged_image, cv2.COLOR_GRAY2BGR)
    bgr2=np.copy(image_bgr)
    edged=np.copy(edged_image)
    Slined_image=draw(edged, bgr2)
    lined_image=draw_lines(edged_image, image_bgr)
    cv2.imshow('EDGES', edged_image)
    cv2.imshow('LINES', lined_image)
    cv2.imshow('standard lines', Slined_image)
main()
