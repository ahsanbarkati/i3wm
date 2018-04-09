import os, glob
import cv2

ulpath = "./shubh_color"

for infile in glob.glob( os.path.join(ulpath, "*.jpg") ):
    im = cv2.imread(infile)
    thumbnail = cv2.CreateMat(100, 100, cv2.CV_8UC3)
    cv2.Resize(im, thumbnail)
    cv2.NamedWindow(infile)
    cv2.ShowImage(infile, thumbnail)
    cv2.WaitKey(0)
    cv2.DestroyWindow(name)