import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

#read input image and find it's edges
original_image=cv2.imread('/home/farman/Documents/my-FYP/Data/board14.jpg')
imgcopy=np.copy(original_image)
imgcopy2=np.copy(original_image)
imgcopy3=np.copy(original_image)
imgcopy4=np.copy(original_image)
gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
image_blur=cv2.bilateralFilter(original_image, 9, sigmaSpace=0, sigmaColor=180)
a=0.66*np.mean(image_blur)
b=2*a
image_edges=cv2.Canny(image_blur,  a, b)
plt.imshow(image_edges),plt.show()

#apply probabilistic hough line transform
lines =cv2.HoughLinesP(image_edges, 0.5, np.pi/360, 50, maxLineGap=100, minLineLength=10)
#print (lines)
if lines is not None:
	for element in lines:
		line= element[0]
		image_lines=cv2.line(imgcopy, (line[0], line[1]), (line[2], line[3]), (0,255,0), 1)
plt.imshow(imgcopy),plt.show()

#apply shi-tomasi corner detector with sub pixel accuracy
corners = cv2.goodFeaturesToTrack(gray,80,0.01,10)
#corners = np.int0(corners)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners2 = cv2.cornerSubPix (gray, corners, (5,5),(-1,-1),criteria)
for i in corners2:
	x,y = i.ravel()
	cv2.circle(imgcopy3, (x,y),1,255,-1)
plt.imshow(imgcopy3),plt.show()	

#compare edges with corners to locate the chessboard
ymax, xmax, _ =original_image.shape
ux, lx = [], []
if lines is not None:
	for element in lines:
		x, y = [], []
		j=0
		x1, y1, x2, y2= element[0]
		for i in corners2:
			x3, y3 = i.ravel()
			d=abs(x3*(y2-y1)-y3*(x2-x1)+x2*y1-y2*x1)/math.sqrt((y2-y1)**2+(x2-x1)**2)
			if d<=2:
				j+=1
				x.append(x3), y.append(y3)
				#print(j)
				if j==6 and x[0] != x[5]:
					x.sort(), y.sort()
					a=(y[0]-y[5])/(x[0]-x[5])
					b=(y[5]*x[0]-y[0]*x[5])/(x[0]-x[5])
					#print(x)
					#print(a, b)
					if (0<=(-b/a) and (-b/a)<=xmax) and ( 0<=((ymax-b)/a) and ((ymax-b)/a<=xmax) ):		#u-l
						cv2.line(imgcopy2, (int(-b/a), 0), (int((ymax-b)/a), ymax), (25,255,10), 1)
						#print(x)
						#print(x[0], y[0], x[5], y[5], -b/a)
						x, y = [], []
						j=0
						ux.append(-b/a), lx.append((ymax-b)/a)
plt.imshow(imgcopy2), plt.title('chess lines'), plt.show()
ux=list(set(ux))
lx=list(set(lx))
lx.sort(), ux.sort()
lxn, uxn =[], []
print(lx)
print(ux)

i=0
for i in range(len(lx)-1):
	ave= ( lx[len(lx)-1] - lx[0] )/(len(lx)-1)
	if abs(lx[i] - lx[i+1]) >= ave/2:
		lxn.append(lx[i])
	if abs(lx[i] - lx[i+1]) < 0.005*xmax:
		lx[i+1]= (lx[i] + lx[i+1])/2
print(ave)


print("the new l is", lxn)

i=0
for i in range(len(ux)-1):
	ave= ( ux[len(ux)-1] - ux[0] )/(len(ux)-1)
	if abs(ux[i] - ux[i+1]) >= ave/2:
		uxn.append(ux[i])
	if abs(ux[i] - ux[i+1]) < 0.005*xmax:
		ux[i+1]= (ux[i] + ux[i+1])/2

print('the new u is', uxn)

i=0
for i in range(len(uxn)):
	cv2.line(imgcopy4, (int(uxn[i]), 0), (int(lxn[i]), ymax), (255, 25, 50), 1)
plt.imshow(imgcopy4), plt.title('chess left lines OOOOO'), plt.show()
i=0
for i in range(5):
	l=len(lxn)-1
	yl=lxn[l] - lxn[l-1]
	xl=lxn[l-1] - lxn[l-2]
	pl=lxn[l] + yl/( 3-4*(yl/(xl+yl)) )
	lxn.append(pl)
	
	yu=uxn[l] - uxn[l-1]
	xu=uxn[l-1] - uxn[l-2]
	pu=uxn[l] + yu/( 3-4*(yu/(xu+yu)) )
	uxn.append(pu)
	print(uxn[len(uxn)-1])
	cv2.line( imgcopy4, (int(uxn[len(uxn)-1]), 0), (int(lxn[len(lxn)-1]), ymax), (255,25,10), 1)
	i+=1
plt.imshow(imgcopy4), plt.title('chess left lines'), plt.show()
i=0
for i in range(20):
	yl=lxn[1] - lxn[0]
	zl=lxn[2] - lxn[1]
	pl=lxn[0] - yl/( 3-4*(yl/(zl+yl)) )
	lxn.insert(0, pl)
	#print(pl)
	
	yu=uxn[1] - uxn[0]
	zu=uxn[2] - uxn[1]
	pu=uxn[0] - yu/( 3-4*(yu/(zu+yu)) )
	uxn.insert(0, pu)
	cv2.line( imgcopy4, (int(uxn[0]), 0), (int(lxn[0]), ymax), (255,25,10), 1)
	i+=1

plt.imshow(imgcopy4), plt.title('chess all lines'), plt.show()
