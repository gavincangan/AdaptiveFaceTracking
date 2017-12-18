import argparse
import dlib
import cv2
import os
import imutils
from os import listdir
from os.path import isfile, join
from cnn_class import Classifier

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def get_pts_in_box(box):
    return (box[0], box[1]), (box[0] + box[2], box[1] + box[3])


ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#	help="path to input image")
ap.add_argument("-i", "--imdir", required=True,
	help="path to input image")
ap.add_argument("-d", "--dest", required=True,
	help="path to save image")
args = vars(ap.parse_args())

model_file = "tf/tensorflow-for-poets-2/tf_files/retrained_graph.pb"
label_file = "tf/tensorflow-for-poets-2/tf_files/retrained_labels.txt"
cnn_classifier = Classifier(model_file=model_file, label_file=label_file)
detector = dlib.get_frontal_face_detector()

mydir = args["imdir"]
dest = args["dest"]
onlyimages = [f for f in listdir(mydir) if isfile(join(mydir, f))]
for imfile in onlyimages:
    if mydir[-1] != '/':
        mydir += '/'
    if dest[-1] != '/':
        dest += '/'
    image = cv2.imread(mydir+imfile)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show the original input image and detect faces in the grayscale
    # image
    # cv2.imshow("Input", image)
    rects = detector(gray, 2)
    for idx, rect in enumerate(rects):
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)
        face = image[y:y + h, x:x + w]
        (label, score) = cnn_classifier.run_data(face)
        print(label, score)
        cv2.imshow('image', face)
        cv2.waitKey(0)

