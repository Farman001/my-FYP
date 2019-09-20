import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

image=cv2.imread('/home/farman/Desktop/demo.jpeg')
#image=cv2.imread('/home/farman/Documents/my-FYP/CALIB/confirmed/IMG_20190224_140532.jpg')
imgcopy=np.copy(image)
imgcopy1=np.copy(image)
imgcopy2=np.copy(image)
imgcopy3=np.copy(image)
ymax, xmax, _ =image.shape
img = np.zeros([ymax,xmax,3],dtype=np.uint8)
img.fill(255)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur=cv2.bilateralFilter(image, 9, sigmaSpace=0, sigmaColor=210)

a=0.66*np.mean(blur)
b=2*a
edged = cv2.Canny(blur, a, b)
#straight lines
lineS =cv2.HoughLinesP(edged, 0.5, np.pi/360, 50, minLineLength=100, maxLineGap=35)
#apply shi-tomasi corner detector with sub pixel accuracy
corners = cv2.goodFeaturesToTrack(gray,80,0.01,10)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners2 = cv2.cornerSubPix (gray, corners, (5,5),(-1,-1),criteria)
for i in corners2:
	x,y = i.ravel()
	cv2.circle(imgcopy1, (x,y),5,255,-1)
plt.imshow(imgcopy1), plt.show()

print('number of lines:', len(lineS))
lines=lineS.tolist()
chess=[]
if lines is not None:
	for element in lines:
		j=0
		x1, y1, x2, y2= element[0]
		for i in corners2:
			x3, y3 = i.ravel()
			d=abs(x3*(y2-y1)-y3*(x2-x1)+x2*y1-y2*x1)/math.sqrt((y2-y1)**2+(x2-x1)**2)
			if d<=5:
				j+=1
				if j==3:
					chess.append(element)
					j=0
					break

print('number of chessboard lines:', len(chess))
ul=[]
ler=[]
if lines is not None:
	for element in lines:
		if element not in chess:
			line= element[0]
			cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0,0,255), 1)
			x0=line[0]
			y0=line[1]
			x1=line[2]
			y1=line[3]
			if x1 != x0 and y1 != y0:
				a= (y1-y0)/(x1-x0)
				b= (y0*x1 - y1*x0)/(x1-x0)
				if (0<=(-b/a) and (-b/a)<=xmax) and ( 0<=((ymax-b)/a) and ((ymax-b)/a<=xmax) ):		#u-l
					cv2.line(imgcopy2, (int(-b/a), 0), (int((ymax-b)/a), ymax), (255,25,10), 2)
					ul.append([-b/a, (ymax-b)/a])
				if ((0<=b) and (b<=ymax)) and ( 0<=(a*xmax+b) and (a*xmax+b)<=ymax ):			#le-r
					cv2.line(imgcopy2, (0, int(b)), (xmax, int(a*xmax+b)), (55,25,210), 2)
					ler.append([b, a*xmax+b])
				if ((0<=b) and (b<=ymax)) and ( 0<=((ymax-b)/a) and ((ymax-b)/a<=xmax) ):		#le-l
					cv2.line(imgcopy2, (0, int(b)),  (int((ymax-b)/a), ymax), (255,25,10), 2)
				if ((0<=b) and (b<=ymax)) and (0<=(-b/a) and (-b/a)<=xmax):				#le-u
					cv2.line(imgcopy2, (int(-b/a), 0), (0, int(b)), (55,25,210), 2)
				if (0<=(-b/a) and (-b/a)<=xmax) and ((a*xmax+b)<=ymax ):				#u-r
					cv2.line(imgcopy2, (int(-b/a), 0), (xmax, int(a*xmax+b)), (255,25,10), 2)
				if ( 0<=((ymax-b)/a) and ((ymax-b)/a<=xmax) ) and ((a*xmax+b)<=ymax ):			#l-r
					cv2.line(imgcopy2, (xmax, int(a*xmax+b)), (int((ymax-b)/a), ymax), (55,25,210), 2)

def takeFirst(elem):
    return elem[0]
ler_n= sorted(ler, key=takeFirst)
ul_n= sorted(ul, key=takeFirst)
up=ler_n[0]
bottom=ler_n[len(ler_n)-1]
left=ul_n[0]
right=ul_n[len(ul_n)-1]
print(ler_n)
"""cv2.line(imgcopy3, (0, int(up[0])), (xmax, int(up[1])), (255,25,10), 5)
cv2.line(imgcopy3, (0, int(bottom[0])), (xmax, int(bottom[1])), (255,25,10), 5)
cv2.line(imgcopy3, (int(left[0]), 0), (int(left[1]), ymax), (55,25,210), 5)
cv2.line(imgcopy3, (int(right[0]), 0), (int(right[1]), ymax), (55,25,210), 5)

cv2.circle(imgcopy3, ( int((br-bb)/(ab-ar)), int( (br-bb)*ar/(ab-ar)+br )), 10, (155,155, 100))
cv2.circle(imgcopy3, ( int((bl-bb)/(ab-al)), int( (bl-bb)*al/(ab-al)+bl )), 10, (155,155, 100))
cv2.circle(imgcopy3, ( int((bl-bu)/(au-al)), int( (bl-bu)*al/(au-al)+bl )), 10, (155,155, 100))
cv2.circle(imgcopy3, ( int((br-bu)/(au-ar)), int( (br-bu)*ar/(au-ar)+br )), 10, (155,155, 100))


"""
ab=(bottom[1]-bottom[0])/xmax
bb=bottom[0]
au=(up[1]-up[0])/xmax
bu=up[0]

ar=ymax/(right[1]-right[0])
br=-ymax*right[0]/(right[1]-right[0])
al=ymax/(left[1]-left[0])
bl=-ymax*left[0]/(left[1]-left[0])

cv2.line(imgcopy3, ( int((br-bb)/(ab-ar)), int( (br-bb)*ar/(ab-ar)+br )), ( int((br-bu)/(au-ar)), int( (br-bu)*ar/(au-ar)+br )), (255,25,10), 8)
cv2.line(imgcopy3, ( int((br-bb)/(ab-ar)), int( (br-bb)*ar/(ab-ar)+br )), ( int((bl-bb)/(ab-al)), int( (bl-bb)*al/(ab-al)+bl )), (255,25,10), 8)
cv2.line(imgcopy3, ( int((br-bu)/(au-ar)), int( (br-bu)*ar/(au-ar)+br )), ( int((bl-bu)/(au-al)), int( (bl-bu)*al/(au-al)+bl )), (255,25,10), 8)
cv2.line(imgcopy3, ( int((bl-bu)/(au-al)), int( (bl-bu)*al/(au-al)+bl )), ( int((bl-bb)/(ab-al)), int( (bl-bb)*al/(ab-al)+bl )), (255,25,10), 8)

plt.imshow(img), plt.show()
plt.imshow(imgcopy3), plt.title('Contours of the object of interest'), plt.show()

my_file=open("/home/farman/AIQ/darknet/data.txt", "w")
my_file.write("%d, %d\n" % ( int((bl-bu)/(au-al)), int( (bl-bu)*al/(au-al)+bl )) )
my_file.write("%d, %d\n" % ( int((br-bu)/(au-ar)), int( (br-bu)*ar/(au-ar)+br )) )
my_file.write("%d, %d\n" % ( int((bl-bb)/(ab-al)), int( (bl-bb)*al/(ab-al)+bl )) )
my_file.write("%d, %d\n" % ( int((br-bb)/(ab-ar)), int( (br-bb)*ar/(ab-ar)+br )) )



my_file.close()
