import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

#read input image and find it's edges
original_image=cv2.imread('/home/farman/Downloads/IMG_20190303_221756.jpg')
imgcopy=np.copy(original_image)
imgcopy2=np.copy(original_image)
imgcopy3=np.copy(original_image)
imgcopy4=np.copy(original_image)
imgcopy5=np.copy(original_image)
gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
image_blur=cv2.GaussianBlur(gray,(5,5),0)
blur= cv2.erode(image_blur, None, iterations=2)
blur = cv2.dilate(image_blur, None, iterations=2)

a=0.66*np.mean(image_blur)
b=2*a
image_edges=cv2.Canny(image_blur,  a, b)
plt.imshow(image_edges),  plt.title('edge lines'), plt.show()

ymax, xmax, _ =original_image.shape
img = np.zeros([ymax,xmax,3],dtype=np.uint8)
img.fill(255)
img1 = np.zeros([ymax,xmax,3],dtype=np.uint8)
img1.fill(255)
####################
contours, hierarchy = cv2.findContours(image_edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
left=xmax
right=0
bottom=0
top=ymax
low=()
up=()
ri=()
lf=()
for c in contours:
	extLeft = tuple(c[c[:, :, 0].argmin()][0])
	extRight = tuple(c[c[:, :, 0].argmax()][0])
	extTop = tuple(c[c[:, :, 1].argmin()][0])
	extBot = tuple(c[c[:, :, 1].argmax()][0])

	if extLeft[0] <= left:
		lf=extLeft
		left=extLeft[0]
	if extRight[0] >= right:
		ri=extRight
		right=extRight[0]
	if extTop[1] <= top:
		up=extTop
		top=extTop[1]
	if extBot[1] >= bottom:
		low=extBot
		bottom=extBot[1]
cv2.line(imgcopy5, lf, up, (0,255,0), 1)
cv2.line(imgcopy5, lf, low, (0,255,0), 1)
cv2.line(imgcopy5, ri, up, (0,255,0), 1)
cv2.line(imgcopy5, low, ri, (0,255,0), 1)
#apply probabilistic hough line transform
lines =cv2.HoughLinesP(image_edges, 0.5, np.pi/360, 50, maxLineGap=100, minLineLength=10)
#print (lines)
if lines is not None:
	for element in lines:
		line= element[0]
		image_lines=cv2.line(imgcopy, (line[0], line[1]), (line[2], line[3]), (0,255,0), 1)
		cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0,0,0), 1)
##############plt.imshow(imgcopy),plt.show()

#apply shi-tomasi corner detector with sub pixel accuracy
corners = cv2.goodFeaturesToTrack(gray,80,0.01,10)
#corners = np.int0(corners)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners2 = cv2.cornerSubPix (gray, corners, (5,5),(-1,-1),criteria)
for i in corners2:
	x,y = i.ravel()
	cv2.circle(imgcopy3, (x,y),1,255,-1)
#############plt.imshow(imgcopy3),plt.show()	

#compare edges with corners to locate the chessboard

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
plt.imshow(imgcopy4), plt.title('averaged chess lines OOOOO'), plt.show()

i=0
minn=uxn[len(uxn)-1]-uxn[0]
for i in range(len(uxn) - 3):
	u1=uxn[i+1] - uxn[i]
	u2=uxn[i+2] - uxn[i+1]
	nu=uxn[i+2] + u2/ (3 - 4*u2/(u2+u1) )
	if minn > abs(uxn[i+3] - nu):
		minn = abs(uxn[i+3] - nu)
		index=i
print('the lowest error for upper side', minn)
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
print('uuuuuuuuuuuuuuuuuuuu', UN)
##############################################################
i=0
minn=lxn[len(lxn)-1]-lxn[0]
for i in range(len(lxn) - 3):
	l1=lxn[i+1] - lxn[i]
	l2=lxn[i+2] - lxn[i+1]
	nl=lxn[i+2] + l2/ (3 - 4*l2/(l2+l1) )
	if minn > abs(lxn[i+3] - nl):
		minn = abs(lxn[i+3] - nl)
		index1=i
print('the lowest error for lower side', minn)
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

print('llllllllllllllllllllllll', LN)

A=index-index1
i=0
if A>0:
	for i in range(30):
		cv2.line(imgcopy5, (int(UN[i]), 0), (int(LN[i+A]), ymax), (255,25,10), 1)
		cv2.line(img, (int(UN[i]), 0), (int(LN[i+A]), ymax), (255,25,10), 1)
else:
	for i in range(30):
		cv2.line(imgcopy5, (int(LN[i]), ymax), (int(UN[i-A]), 0), (255,25,10), 1)
		cv2.line(img, (int(LN[i]), ymax), (int(UN[i-A]), 0), (255,25,10), 1)
plt.imshow(imgcopy5), plt.title('bombastic'), plt.show()


j=0
k=0
for j in range(len(LN)):
	a1=(UN[j] - LN[j]) / (0-xmax)
	b1=UN[j]
	

	i=0
	for i in range(len(lines)):
		a2=(lines[i][0][1] - lines[i][0][3]) / (lines[i][0][0] - lines[i][0][2])
		b2=(-lines[i][0][1]*lines[i][0][2] + lines[i][0][3]*lines[i][0][0]) / (lines[i][0][0] - lines[i][0][2])

		xi=(b2-b1)/(a1-a2)
		yi=(b2*a1-a2*b1)/(a1-a2)
		if (min(lines[i][0][0], lines[i][0][2]) <= xi and xi <= max(lines[i][0][0], lines[i][0][2])) and (min(lines[i][0][1], lines[i][0][3]) <= yi and yi <= max(lines[i][0][1], lines[i][0][3])):
			img1[int(yi), int(xi)] = (0,0,10)
			k+=1
print(k)
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
plt.imshow(img), plt.title('border lines vs chess lines'), plt.show()
plt.imshow(img1), plt.title('intersection points of edge lines with chessboard lines'), plt.show()
i=0
for i in range(20 ):
	l=len(lxn)-1
	yl=lxn[l] - lxn[l-1]
	xl=lxn[l-1] - lxn[l-2]
	pl=lxn[l] + yl/( 3-4*(yl/(xl+yl)) )
	lxn.append(pl)
	print(lxn[len(lxn)-1])
	
	yu=uxn[l] - uxn[l-1]
	xu=uxn[l-1] - uxn[l-2]
	pu=uxn[l] + yu/( 3-4*(yu/(xu+yu)) )
	uxn.append(pu)
	print(uxn[len(uxn)-1])
	print('/////////////////')
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
