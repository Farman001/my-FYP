import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

#read input image and find it's edges
original_image=cv2.imread('/home/farman/Documents/my-FYP/chess6.png')
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

#apply shi-tomasi corner detector
corners = cv2.goodFeaturesToTrack(gray,80,0.01,10)
corners = np.int0(corners)
for i in corners:
	x,y = i.ravel()
	cv2.circle(imgcopy3, (x,y),1,255,-1)
plt.imshow(imgcopy3),plt.show()	

#compare edges with corners to locate the chessboard
ymax, xmax, _ =original_image.shape
print(xmax, ymax)
Ux, Lx, Ley, Ry =[], [], [], []
A, B= [], []
i, j = 0, 0
if lines is not None:
	for element in lines:
		x1, y1, x2, y2= element[0]
		for i in corners:
			x3, y3 = i.ravel()
			d=abs(x3*(y2-y1)-y3*(x2-x1)+x2*y1-y2*x1)/math.sqrt((y2-y1)**2+(x2-x1)**2)
			if d<=2 :
				j+=1
				if j==6 and x1 != x2:
					a=(y1-y2)/(x1-x2)
					b=(y2*x1-y1*x2)/(x1-x2)
					A.append(a), B.append(b)
					if (0<=(-b/a) and (-b/a)<=xmax) and (0<=b and b<=ymax):						#u-le
						cv2.line(imgcopy2, (int(-b/a), 0), (0, int(b)), (35,255,0), 1)

					elif (0<=(-b/a) and (-b/a)<=xmax) and ( 0<=((ymax-b)/a) and ((ymax-b)/a<=xmax) ):		#u-l
						cv2.line(imgcopy2, (int(-b/a), 0), (int((ymax-b)/a), ymax), (135,255,0), 1)
						Ux.append(int(-b/a)), Lx.append(int((ymax-b)/a))

					elif (0<=(-b/a) and (-b/a)<=xmax) and ( 0<=a*xmax+b and a*xmax+b<=ymax ):			#u-r
						cv2.line(imgcopy2, (int(-b/a), 0), (xmax, int(a*xmax+b)), (235,255,0), 1)
						Ux.append(int(-b/a)), Lx.append(int((ymax-b)/a))

					elif (0<=b and b<=ymax) and ( 0<=((ymax-b)/a) and ((ymax-b)/a<=xmax) ):				#le-l
						cv2.line(imgcopy2, (0, int(b)), (int((ymax-b)/a), ymax), (205, 35,0), 1)

					elif (0<=b and b<=ymax) and ( 0<=a*xmax+b and a*xmax+b<=ymax ):					#le-r
						cv2.line(imgcopy2, (0, int(b)), (xmax, int(a*xmax+b)), (205,155,0), 1)
						Ley.append(int(b)), Ry.append(int(a*xmax+b))

					else:												#l-r
						cv2.line(imgcopy2, (int((ymax-b)/a), ymax), (xmax, int(a*xmax+b)), (55,25,255), 1)
					j=0
plt.imshow(imgcopy2), plt.title('chess'), plt.show()
Lx.sort()
Ux.sort()
print(Ux)
Lxn, Uxn =[], []
i=0

for i in range(len(Lx)-2):
	ave= ( Lx[len(Lx)-1] - Lx[0] )/len(Lx)
	if abs(Lx[i] - Lx[i+1]) >= ave/2:
		Lxn.append(Lx[i])
print(Lxn, len(Lxn))
print(ave)

for i in range(len(Ux)-2):
	ave= ( Ux[len(Ux)-1] - Ux[0] )/len(Ux)
	if abs(Ux[i] - Ux[i+1]) >= ave/2:
		Uxn.append(Ux[i])
print(Uxn, len(Uxn))
print(ave)

j=0
for j in range(len(Uxn)):
	cv2.line( imgcopy4, (Uxn[j], 0), (Lxn[j], ymax), (255,25,10), 1)
for i in range(20):
	l=len(Lxn)-1
	yl=Lxn[l] - Lxn[l-1]
	xl=Lxn[l-1] - Lxn[l-2]
	pl=Lxn[l] + yl/( 3-4*(yl/(xl+yl+0.3)) )
	Lxn.append(int(pl))
	
	yu=Uxn[l] - Uxn[l-1]
	xu=Uxn[l-1] - Uxn[l-2]
	pu=Uxn[l] + yu/( 3-4*(yu/(xu+yu+0.3)) )
	Uxn.append(int(pu))
	cv2.line( imgcopy4, (int(Uxn[len(Uxn)-1]), 0), (int(Lxn[len(Lxn)-1]), ymax), (255,25,10), 1)
	i+=1
i=0
for i in range(20):
	yl=Lxn[1] - Lxn[0]
	zl=Lxn[2] - Lxn[1]
	pl=Lxn[0] - yl/( 3-4*(yl/(zl+yl+0.3)) )
	Lxn.insert(0, int(pl))
	
	yu=Uxn[1] - Uxn[0]
	zu=Uxn[2] - Uxn[1]
	pu=Uxn[0] - yu/( 3-4*(yu/(zu+yu+0.3)) )
	Uxn.insert(0, int(pu))
	cv2.line( imgcopy4, (int(Uxn[0]), 0), (int(Lxn[0]), ymax), (255,25,10), 1)
	i+=1
plt.imshow(imgcopy4), plt.title('chess'), plt.show()
