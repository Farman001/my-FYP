import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 25, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
x, y = [], []
for i in range(6):
	for j in range(9):
		x.append(25*j), y.append(25*i)
nx=np.asarray(x)
ny=np.asarray(y)
objp=np.hstack((nx.reshape(54,1),ny.reshape(54,1),np.zeros((54,1)))).astype(np.float32)
print(objp)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('./CALIB/confirmed/*.jpg')

npatternfound=0
imgNotGood=images[1]

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print('reading the image', fname)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,6),None)


    # If found, add object points, image points (after refining them)
    if ret == True:
       	print('pattern found')
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        img = cv.drawChessboardCorners(img, (9,6), corners2,ret)
        #plt.imshow( img)
        #plt.show()
        #k=cv.waitKey(0) & 0xFF
        #if k==25:
        #	print('image skipped')
        #	imgNotGood=fname
        #	continue
        print('image accepted')
        npatternfound+=1
        objpoints.append(objp)
        imgpoints.append(corners)
    else:
        imgNotGood=fname

cv.destroyAllWindows()

if npatternfound>1:
	print(npatternfound, " good images found")
	ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
	img = cv.imread('/home/farman/Downloads/object1.jpg')
	h,  w = img.shape[:2]
	newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
	mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
	dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

	# crop the image
	x, y, w, h = roi
	print(roi)
	dst = dst[y:y+h, x:x+w]
	cv.imwrite('calibresult.png', dst)
	print(ret)
	print('calibration matrix')
	print(mtx)
	print('distortion coefficients')
	print(dist)
	total_error = 0
	for i in range(len(objpoints)):
	    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
	    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
	    total_error += error
	print( "mean error: {}".format(total_error/len(objpoints)) )

print('rotational verstors:', rvecs)
print('???????????????????????????????????????????????')
print('translational vectors', tvecs)
