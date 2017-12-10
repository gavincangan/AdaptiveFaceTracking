# USAGE
# Adding bulk align capabilities to run against large datasets
# python align_faces.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg


# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import os
from os import listdir
from os.path import isfile, join

def align_image(image, fa, imname="img", dest="./results/"):
    # load the input image, resize it, and convert it to grayscale
    print('Processing', imname)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show the original input image and detect faces in the grayscale
    # image
    # cv2.imshow("Input", image)
    rects = detector(gray, 2)

    # loop over the face detections
    for idx, rect in enumerate(rects):
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)
        if (w < 10 or h < 0):
            print('Skipping', imname)
            continue
        try:
            faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
            faceAligned = fa.align(image, gray, rect)
        except:
            print('Skipping', imname)
            continue

        # import uuid
        # f = str(uuid.uuid4())
        # path = "./results/" + os.path.splitext(imname)[0] + "/" # + str(idx) + ".png"
        path = dest
        if not os.path.exists(path):
                os.makedirs(path)
        path = path + os.path.splitext(imname)[0] + "_" + str(idx) + ".jpg"
        print("Saving", path)
        cv2.imwrite(path, faceAligned)

        # display the output images
        #cv2.imshow("Original", faceOrig)
        #cv2.imshow("Aligned", faceAligned)
        #cv2.waitKey(0)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
#ap.add_argument("-i", "--image", required=True,
#	help="path to input image")
ap.add_argument("-i", "--imdir", required=True,
	help="path to input image")
ap.add_argument("-d", "--dest", required=True,
	help="path to save image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=256)
mydir = args["imdir"]
dest = args["dest"]
onlyimages = [f for f in listdir(mydir) if isfile(join(mydir, f))]
for imfile in onlyimages:
    if mydir[-1] != '/':
        mydir += '/'
    if dest[-1] != '/':
        dest += '/'
    image = cv2.imread(mydir+imfile)
    align_image(image, fa, imfile, dest)
#cv2.waitKey(0)
