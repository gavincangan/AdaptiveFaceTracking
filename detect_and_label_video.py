import argparse
import dlib
import cv2
import os
import imutils
from os import listdir
from os.path import isfile, join
from cnn_class import Classifier
from math import ceil
import sys

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale = 1
lineType = 2

# face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
profile_cascade = cv2.CascadeClassifier('data/haarcascade_profileface.xml')

frontal_face_dir = './face_images/front_'
side_profile_face_dir = './face_images/prof_'

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

def rects_overlap(rect1, rect2):
    cent1x = rect1[0]+float(rect1[2])/2
    cent1y = rect1[1]+float(rect1[3])/2
    cent2x = rect2[0]+float(rect2[2])/2
    cent2y = rect2[1]+float(rect2[3])/2
    if abs(cent1x-cent2x) < float(rect1[2])/2 or \
            abs(cent1y-cent2y) < float(rect2[3])/2:
        return True
    else:
        return False


def scale_rect(rect, scale):
    # print(rect)
    border = int(ceil(rect[3]*(scale-1)))
    # print(border)
    new_rect = [rect[0]-border, rect[1]-border, rect[2]+2*border, rect[3]+2*border]
    # print(new_rect)
    box_oor = False
    for pt in new_rect:
        if(pt < 0 or pt > 800):
            box_oor = True
    # print(new_rect)
    if box_oor:
        new_rect = rect
    return new_rect

ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#	help="path to input image")
args = vars(ap.parse_args())

model_file = "tf/tensorflow-for-poets-2/tf_files/retrained_graph.pb"
label_file = "tf/tensorflow-for-poets-2/tf_files/retrained_labels.txt"
cnn_classifier = Classifier(model_file=model_file, label_file=label_file)
detector = dlib.get_frontal_face_detector()


# Read video
video = cv2.VideoCapture("./TBBT_S10E16.mp4")

# Exit if video not opened.
if not video.isOpened():
    print "Could not open video"
    sys.exit()

# Read first frame.
ok, frame = video.read()
if not ok:
    print 'Cannot read video file'
    sys.exit()

cur_frame = 0
while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    image = frame
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show the original input image and detect faces in the grayscale
    # image
    # cv2.imshow("Input", image)
    # rects = detector(gray, 2)
    front_faces = face_cascade.detectMultiScale(gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(10, 10),
            flags = (cv2.CASCADE_SCALE_IMAGE +
                cv2.CASCADE_DO_CANNY_PRUNING +
                cv2.CASCADE_FIND_BIGGEST_OBJECT +
                cv2.CASCADE_DO_ROUGH_SEARCH))

    front_faces = [scale_rect(f, 1.2) for f in front_faces]

    profile_faces = profile_cascade.detectMultiScale(gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(10, 10),
            flags = (cv2.CASCADE_SCALE_IMAGE +
                cv2.CASCADE_DO_CANNY_PRUNING +
                cv2.CASCADE_FIND_BIGGEST_OBJECT +
                cv2.CASCADE_DO_ROUGH_SEARCH))

    profile_faces = [scale_rect(p, 1.2) for p in profile_faces]

    for f in front_faces:
        profile_faces = [p for p in profile_faces if not rects_overlap(f,p)]

    for idx, rect in enumerate(front_faces):
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        # (x, y, w, h) = rect_to_bb(rect)
        x = rect[0]
        y = rect[1]
        w = rect[2]
        h = rect[3]
        face = image[y:y + h, x:x + w]
        cv2.imwrite(frontal_face_dir + str(cur_frame) + str(idx) + '.jpg', face)
        (label, score) = cnn_classifier.run_data(face)
        # print(label, score)
        # cv2.imshow('image', image)
        # cv2.imshow('face', face)
        [this_pt1, this_pt2] = get_pts_in_box((x, y, w, h))
        cv2.rectangle(image, this_pt1, this_pt2, (0,255,255))
        cv2.putText(image, label, this_pt1, font, fontScale,
                (0,255,255), lineType)


    for idx, rect in enumerate(profile_faces):
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        # (x, y, w, h) = rect_to_bb(rect)
        x = rect[0]
        y = rect[1]
        w = rect[2]
        h = rect[3]
        face = image[y:y + h, x:x + w]
        cv2.imwrite(side_profile_face_dir + str(cur_frame)  + str(idx) + '.jpg', face)
        (label, score) = cnn_classifier.run_data(face)
        # print(label, score)
        # cv2.imshow('image', image)
        # cv2.imshow('face', face)
        [this_pt1, this_pt2] = get_pts_in_box((x, y, w, h))
        cv2.rectangle(image, this_pt1, this_pt2, (0,255,0))
        cv2.putText(image, label, this_pt1, font, fontScale,
                (0,255,0), lineType)


    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    print('saving', cur_frame);
    cv2.imwrite('video_out/'+str(cur_frame)+'.jpg', image)
    cur_frame += 1
    if (cur_frame > 2000):
        break

