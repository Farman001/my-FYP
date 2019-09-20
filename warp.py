import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

original_image=cv2.imread('/home/farman/Desktop/demo.jpeg')

ymax, xmax, _ =original_image.shape

my_file=open("/home/farman/AIQ/darknet/data.txt", "r")
lines=my_file.readlines()
tl=lines[0].strip().split(", ")
tr=lines[1].strip().split(", ")
bl=lines[2].strip().split(", ")
br=lines[3].strip().split(", ")


pts1 = np.float32([ [tl[0], tl[1]], [tr[0], tr[1]], [bl[0], bl[1]], [br[0], br[1]] ])
pts2 = np.float32([[0,0],[1000,0],[0,800],[1000,800]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(original_image,M,(1000,800))

plt.imshow(dst), plt.title('The part of image with object of interst is extracted'), plt.show()
cv2.imwrite('warpresult.png', dst)
