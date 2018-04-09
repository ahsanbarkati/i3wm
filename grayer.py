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

ulpath = pos
c=0
for infile in glob.glob( os.path.join(ulpath, "*.jpg") ):
    im = cv2.imread(infile)
    resized_image = cv2.resize(im, (50, 50))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('positives/'+str(c)+'.jpg',gray_image)
    c=c+1

ulpath = neg
c=0
for infile in glob.glob( os.path.join(ulpath, "*.jpg") ):
    im = cv2.imread(infile)
    resized_image = cv2.resize(im, (50, 50))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('negatives/'+str(c)+'.jpg',gray_image)
    c=c+1
