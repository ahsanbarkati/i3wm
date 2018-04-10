import os, glob
import cv2
import sys

pos=sys.argv[1]
neg=sys.argv[2]
try:
	os.system("mkdir positives")
	os.system('mkdir negatives')
except:
	print('Directory already exists')

haar= cv2.CascadeClassifier('Face_cascade.xml')
ulpath = pos
c=0
dim=400
for infile in glob.glob( os.path.join(ulpath, "*.jpg") ):
	im = cv2.imread(infile)
	resized_image = cv2.resize(im, (dim, dim))
	im = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
	faces=haar.detectMultiScale(im,scaleFactor=1.1,minNeighbors=5)
	if(len(faces)):
		[x,y,w,h]=faces[0]
		im=im[y-10:y+h+10,x-10:x+w+10]
	im = cv2.resize(im,None,fx=10, fy=10, interpolation = cv2.INTER_LINEAR)
	cv2.imwrite('positives/'+str(c)+'.jpg',im)
	c=c+1

ulpath = neg
c=0
for infile in glob.glob( os.path.join(ulpath, "*.jpg") ):
	im = cv2.imread(infile)
	resized_image = cv2.resize(im, (dim, dim))
	im = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
	faces=haar.detectMultiScale(im,scaleFactor=1.1,minNeighbors=5)
	if(len(faces)):
		[x,y,w,h]=faces[0]
		im=im[y-10:y+h+10,x-10:x+w+10]
	im = cv2.resize(im,None,fx=10, fy=10, interpolation = cv2.INTER_LINEAR)
	im = cv2.resize(im, (dim, dim))

	cv2.imwrite('negatives/'+str(c)+'.jpg',im)
	c=c+1
