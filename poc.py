import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

#read input image and find it's edges
#original_image=cv2.imread('/home/farman/Downloads/IMG_20190224_162113.jpg')
#original_image=cv2.imread('/home/farman/Downloads/IMG_20190207_013347.jpg')
original_image=cv2.imread('/home/farman/Documents/my-FYP/warpresult.png')
imgcopy=np.copy(original_image)
imgcopy2=np.copy(original_image)
imgcopy3=np.copy(original_image)
imgcopy4=np.copy(original_image)
imgcopy5=np.copy(original_image)
imgcopy6=np.copy(original_image)
gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
image_blur=cv2.GaussianBlur(gray,(5,5),0)

a=0.66*np.mean(image_blur)
b=2*a
image_edges=cv2.Canny(image_blur,  a, b)
plt.imshow(image_edges),  plt.title('Edges detected by Canny detector'), plt.show()
ymax, xmax, _ =original_image.shape
img = np.zeros([ymax,xmax,3],dtype=np.uint8)
img.fill(255)
img1 = np.zeros([ymax,xmax,3],dtype=np.uint8)
img1.fill(255)
#apply probabilistic hough line transform
lines =cv2.HoughLinesP(image_edges, 0.5, np.pi/360, 50, maxLineGap=100, minLineLength=10)
#print (lines)
if lines is not None:
	for element in lines:
		line= element[0]
		cv2.line(img, (line[0], line[1]), (line[2], line[3]), (155,55,105), 2)
plt.imshow(img), plt.title('All line segments detected by Probabilistic Hough Line Transform'), plt.show()
#apply shi-tomasi corner detector with sub pixel accuracy
corners = cv2.goodFeaturesToTrack(gray,80,0.01,10)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners2 = cv2.cornerSubPix (gray, corners, (5,5),(-1,-1),criteria)
for i in corners2:
	x,y = i.ravel()
	cv2.circle(img1, (x,y),7,255,-1)
plt.imshow(img1), plt.title('All corners detected by Shi-Tomasi detector'), plt.show()
#compare edges with corners to locate the chessboard
ux, lx = [], []
ley, ry = [], []
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
				if j==5 and x[0] != x[4]:
					#x.sort(), y.sort()
					a=(y[0]-y[4])/(x[0]-x[4])
					b=(y[4]*x[0]-y[0]*x[4])/(x[0]-x[4])
					#print(x)
					#print(a, b)
					if (0<=(-b/a) and (-b/a)<=xmax) and ( 0<=((ymax-b)/a) and ((ymax-b)/a<=xmax) ):		#u-l
						cv2.line(imgcopy2, (int(-b/a), 0), (int((ymax-b)/a), ymax), (255,25,10), 1)
						x, y = [], []
						j=0
						ux.append(-b/a), lx.append((ymax-b)/a)
					if ((0<=b) and (b<=ymax)) and ( 0<=(a*xmax+b) and (a*xmax+b)<=ymax ):			#le-r
						cv2.line(imgcopy2, (0, int(b)), (xmax, int(a*xmax+b)), (55,25,210), 1)
						x, y = [], []
						j=0
						ley.append(b), ry.append(a*xmax+b)
					if ((0<=b) and (b<=ymax)) and ( 0<=((ymax-b)/a) and ((ymax-b)/a<=xmax) ):		#le-l
						cv2.line(imgcopy2, (0, int(b)),  (int((ymax-b)/a), ymax), (255,25,10), 1)
						x, y = [], []
						j=0
						ux.append(b), lx.append((ymax-b)/a)
plt.imshow(imgcopy2), plt.title('Chessboard lines elongated throughout the image'), plt.show()
##############################
ux=list(set(ux))
lx=list(set(lx))
lx.sort(), ux.sort()
lxn, uxn =[], []
i=0
for i in range(len(lx)-1):
	ave= ( lx[len(lx)-1] - lx[0] )/(len(lx)-1)
	if abs(lx[i] - lx[i+1]) >= ave/2:
		lxn.append(lx[i])
	if abs(lx[i] - lx[i+1]) < 0.005*xmax:
		lx[i+1]= (lx[i] + lx[i+1])/2
i=0
for i in range(len(ux)-1):
	ave= ( ux[len(ux)-1] - ux[0] )/(len(ux)-1)
	if abs(ux[i] - ux[i+1]) >= ave/2:
		uxn.append(ux[i])
	if abs(ux[i] - ux[i+1]) < 0.005*xmax:
		ux[i+1]= (ux[i] + ux[i+1])/2
for i in range(len(uxn)):
	cv2.line(imgcopy4, (int(uxn[i]), 0), (int(lxn[i]), ymax), (255,25,10), 1)
######################################
ley=list(set(ley))
ry=list(set(ry))
ley.sort(), ry.sort()
leyn, ryn =[], []
i=0
for i in range(len(ley)-1):
	ave= ( ley[len(ley)-1] - ley[0] )/(len(ley)-1)
	if abs(ley[i] - ley[i+1]) >= ave/2:
		leyn.append(ley[i])
	if abs(ley[i] - ley[i+1]) < 0.005*ymax:
		ley[i+1]= (ley[i] + ley[i+1])/2
i=0
for i in range(len(ry)-1):
	ave= ( ry[len(ry)-1] - ry[0] )/(len(ry)-1)
	if abs(ry[i] - ry[i+1]) >= ave/2:
		ryn.append(ry[i])
	if abs(ry[i] - ry[i+1]) < 0.005*ymax:
		ry[i+1]= (ry[i] + ry[i+1])/2
for i in range(len(ryn)):
	cv2.line(imgcopy4, (0, int(leyn[i])), (xmax, int(ryn[i])), (55,25,210), 1)
plt.imshow(imgcopy4), plt.title('Averaged Chessboard lines elongated throughout the image'), plt.show()
###############################################################################################
avele=(leyn[len(leyn)-1]-leyn[0])/(len(leyn)-1)
aver=(ryn[len(ryn)-1]-ryn[0])/(len(ryn)-1)

for i in range(10):
	cv2.line(imgcopy4, (0, int(leyn[0]-avele*i)), (1000, int(ryn[0]-aver*i)), (55,25,210), 3)
	cv2.line(imgcopy4, (0, int(leyn[len(leyn)-1]+avele*i)), (1000, int(ryn[len(ryn)-1]+aver*i)), (55,25,210), 3)

aveu=(uxn[len(uxn)-1]-uxn[0])/(len(uxn)-1)
avel=(lxn[len(lxn)-1]-lxn[0])/(len(lxn)-1)
for i in range(10):
	cv2.line(imgcopy4, (int(uxn[0]-aveu*i), 0), (int(lxn[0]-avel*i), ymax), (255,25,10), 3)
	cv2.line(imgcopy4, (int(uxn[len(uxn)-1]+aveu*i), 0), (int(lxn[len(lxn)-1]+avel*i), ymax), (255,25,10), 3)
plt.imshow(imgcopy4), plt.title('Chessboard lines are spreaded over the entire image'), plt.show()

length=(1000/avel)*25
width=(800/avele)*25
top=(1000/aveu)*25
right=(800/aver)*25
print('length of bottom side is:', length, 'mm')
print('length of top side is:', top, 'mm')
print('length of left side is:', width, 'mm')
print('length of right side is:', right, 'mm')
print('surface area is :', top*width, 'mm2')
