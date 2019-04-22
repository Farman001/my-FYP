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
						cv2.line(imgcopy2, (int(-b/a), 0), (int((ymax-b)/a), ymax), (255,25,10), 2)
						x, y = [], []
						j=0
						ux.append(-b/a), lx.append((ymax-b)/a)
					if ((0<=b) and (b<=ymax)) and ( 0<=(a*xmax+b) and (a*xmax+b)<=ymax ):			#le-r
						cv2.line(imgcopy2, (0, int(b)), (xmax, int(a*xmax+b)), (55,25,210), 2)
						x, y = [], []
						j=0
						ley.append(b), ry.append(a*xmax+b)
					if ((0<=b) and (b<=ymax)) and ( 0<=((ymax-b)/a) and ((ymax-b)/a<=xmax) ):		#le-l
						cv2.line(imgcopy2, (0, int(b)),  (int((ymax-b)/a), ymax), (255,25,10), 2)
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
	cv2.line(imgcopy4, (int(uxn[i]), 0), (int(lxn[i]), ymax), (255,25,10), 3)
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
	cv2.line(imgcopy4, (0, int(leyn[i])), (xmax, int(ryn[i])), (55,25,210), 3)
plt.imshow(imgcopy4), plt.title('Averaged Chessboard lines elongated throughout the image'), plt.show()
##############################################
i=0
minu=uxn[len(uxn)-1]-uxn[0]
for i in range(len(uxn) - 3):
	u1=uxn[i+1] - uxn[i]
	u2=uxn[i+2] - uxn[i+1]
	nu=uxn[i+2] + u2/ (3 - 4*u2/(u2+u1) )
	if minu > abs(uxn[i+3] - nu):
		minu = abs(uxn[i+3] - nu)
		index=i
print('the lowest error for upper side', minu)
i=0
xu=uxn[index+1] - uxn[index]
yu=uxn[index+2] - uxn[index+1]
zu=yu/( 3- 4*yu/(xu+yu) )
u3=uxn[index+2]
UN=[uxn[index], uxn[index+1], uxn[index+2] ]
for i in range(15):
	xu=yu
	yu=zu
	u3=u3 + zu
	UN.append(u3)
	zu=yu/( 3- 4*yu/(xu+yu) )

i=0
yu=uxn[index+1] - uxn[index]
zu=uxn[index+2] - uxn[index+1]
xu=yu/( 3- 4*yu/(zu+yu) )
u0=uxn[index]
for i in range(15):
	zu=yu
	yu=xu
	u0=u0 - xu
	UN.insert(0, u0)
	xu=yu/( 3- 4*yu/(zu+yu) )
#print('uuuuuuuuuuuuuuuuuuuu', UN)
#|||||||||||||||||||||||||||||||||||||||||||
i=0
minl=lxn[len(lxn)-1]-lxn[0]
for i in range(len(lxn) - 3):
	l1=lxn[i+1] - lxn[i]
	l2=lxn[i+2] - lxn[i+1]
	nl=lxn[i+2] + l2/ (3 - 4*l2/(l2+l1) )
	if minl > abs(lxn[i+3] - nl):
		minl = abs(lxn[i+3] - nl)
		index1=i
print('the lowest error for lower side', minl)
i=0
xl=lxn[index1+1] - lxn[index1]
yl=lxn[index1+2] - lxn[index1+1]
zl=yl/( 3- 4*yl/(xl+yl) )
l3=lxn[index1+2]
LN=[lxn[index1], lxn[index1+1], lxn[index1+2] ]
for i in range(15):
	xl=yl
	yl=zl
	l3=l3 + zl
	LN.append(l3)
	zl=yl/( 3- 4*yl/(xl+yl) )
i=0
yl=lxn[index1+1] - lxn[index1]
zl=lxn[index1+2] - lxn[index1+1]
xl=yl/( 3- 4*yl/(zl+yl) )
l0=lxn[index1]
for i in range(15):
	zl=yl
	yl=xl
	l0=l0 - xl
	LN.insert(0, l0)
	xl=yl/( 3- 4*yl/(zl+yl) )

#print('llllllllllllllllllllllll', LN)
A=index-index1
i=0
if A>0:
	for i in range(30):
		cv2.line(imgcopy5, (int(UN[i]-10), 0), (int(LN[i+A]), ymax), (255,25,10), 3)
else:
	for i in range(30):
		cv2.line(imgcopy5, (int(UN[i-A]-10), 0), (int(LN[i]), ymax), (255,25,10), 3)
plt.imshow(imgcopy5), plt.title(''), plt.show()
############################################################
i=0
minn=leyn[len(leyn)-1]-leyn[0]
for i in range(len(leyn) - 3):
	u1=leyn[i+1] - leyn[i]
	u2=leyn[i+2] - leyn[i+1]
	nu=leyn[i+2] + u2/ (3 - 4*u2/(u2+u1) )
	if minn > abs(leyn[i+3] - nu):
		minn = abs(leyn[i+3] - nu)
		index=i
print('the lowest error for left side', minn)
i=0
xu=leyn[index+1] - leyn[index]
yu=leyn[index+2] - leyn[index+1]
zu=yu/( 3- 4*yu/(xu+yu) )
u3=leyn[index+2]
LE=[leyn[index], leyn[index+1], leyn[index+2] ]
for i in range(15):
	xu=yu
	yu=zu
	u3=u3 + zu
	LE.append(u3)
	zu=yu/( 3- 4*yu/(xu+yu) )

i=0
yu=leyn[index+1] - leyn[index]
zu=leyn[index+2] - leyn[index+1]
xu=yu/( 3- 4*yu/(zu+yu) )
u0=leyn[index]
for i in range(25):
	zu=yu
	yu=xu
	u0=u0 - xu
	LE.insert(0, u0)
	xu=yu/( 3- 4*yu/(zu+yu) )
#print('LEFTTTTTTTTTTTTTTTTTTTTTTTTTTT', LE)
#|||||||||||||||||||||||||||||||||||||||||||
i=0
minn=ryn[len(ryn)-1]-ryn[0]
for i in range(len(ryn) - 3):
	l1=ryn[i+1] - ryn[i]
	l2=ryn[i+2] - ryn[i+1]
	nl=ryn[i+2] + l2/ (3 - 4*l2/(l2+l1) )
	if minn > abs(ryn[i+3] - nl):
		minn = abs(ryn[i+3] - nl)
		index1=i
print('the lowest error for right side', minn)
i=0
xl=ryn[index1+1] - ryn[index1]
yl=ryn[index1+2] - ryn[index1+1]
zl=yl/( 3- 4*yl/(xl+yl) )
l3=ryn[index1+2]
RN=[ryn[index1], ryn[index1+1], ryn[index1+2] ]
for i in range(25):
	xl=yl
	yl=zl
	l3=l3 + zl
	RN.append(l3)
	zl=yl/( 3- 4*yl/(xl+yl) )
i=0
yl=ryn[index1+1] - ryn[index1]
zl=ryn[index1+2] - ryn[index1+1]
xl=yl/( 3- 4*yl/(zl+yl) )
l0=ryn[index1]
for i in range(25):
	zl=yl
	yl=xl
	l0=l0 - xl
	RN.insert(0, l0)
	xl=yl/( 3- 4*yl/(zl+yl) )
#print('RIGHTTTTTTTTTTTTTTTTTTTTTTTT', RN)
A=index-index1
i=0
if A>0:
	for i in range(37):
		cv2.line(imgcopy5, (0, int(LE[i-A])), (xmax, int(RN[i])), (55,25,210), 3)
else:
	for i in range(37):
		cv2.line(imgcopy5, (0, int(LE[i-A])), (xmax, int(RN[i])), (55,25,210), 3)



my_file=open("/home/farman/AIQ/darknet/data.txt", "r")
lines=my_file.readlines()
tl=lines[0].strip().split(", ")
tr=lines[1].strip().split(", ")
bl=lines[2].strip().split(", ")
br=lines[3].strip().split(", ")

cv2.line(imgcopy5, (int(tl[0]), int(tl[1])), (int(bl[0]), int(bl[1])), (155,125,20), 7)
cv2.line(imgcopy5, (int(tl[0]), int(tl[1])), (int(tr[0]), int(tr[1])), (155,125,20), 7)
cv2.line(imgcopy5, (int(br[0]), int(br[1])), (int(bl[0]), int(bl[1])), (155,125,20), 7)
cv2.line(imgcopy5, (int(br[0]), int(br[1])), (int(tr[0]), int(tr[1])), (155,125,20), 7)


plt.imshow(imgcopy5), plt.title('Paralel lines to chesboard on all over the image'), plt.show()
