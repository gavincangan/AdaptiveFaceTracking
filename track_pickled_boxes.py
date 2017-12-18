import pickle
import cv2
import glob
from os import listdir
from os.path import isfile, join
import numpy as np

from cnn_class import Classifier

# home_folder = '/home/gavincangan/computerVision/AdaptiveFaceTracking/'
home_folder = './'

model_file = "tf/tensorflow-for-poets-2/tf_files/retrained_graph.pb"
label_file = "tf/tensorflow-for-poets-2/tf_files/retrained_labels.txt"
cnn_classifier = Classifier(model_file=model_file, label_file=label_file)

box_in_frame_file = home_folder + 'box_in_frame.pick'
person_in_frame_box_file = home_folder + 'person_in_frame_box.pick'
tracked_next_frame_box_file = home_folder + 'tracked_next_frame_box.pick'

box_in_frame_fp = open(box_in_frame_file, 'r+b')
person_in_frame_box_fp = open(person_in_frame_box_file, 'r+b')
tracked_next_frame_box_fp = open(tracked_next_frame_box_file, 'r+b')

box_in_frame = pickle.load(box_in_frame_fp)
person_in_frame_box = pickle.load(person_in_frame_box_fp)
tracked_next_frame_box = pickle.load(tracked_next_frame_box_fp)

frames_folder = home_folder + 'frames'
boxes_folder = home_folder + 'pickled_boxes'
output_frames_folder = home_folder + 'output_frames'

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale = 1
lineType = 2
personColors = ((128, 0, 0),
        (0, 128, 0),
        (128, 128, 0),
        (0, 0, 128),
        (128, 0, 128),
        (0, 128, 128),
        (128, 128, 128),
        (255, 0, 0),
        (0, 255, 0),
        (255, 255, 0),
        (0, 0, 255),
        (255, 0, 255))

INVALID = -1
LEONARD = 1
PENNY = 2
SHELDON = 3
HOWARD = 4
RAJ = 5
BERNADETTE = 6
AMY = 7

faces = [LEONARD, PENNY, SHELDON, HOWARD, RAJ, BERNADETTE, AMY]
face_tracker = dict()

start_frame_num = 80
end_frame_num = 24499


def get_name_string(person_id):
    if person_id == LEONARD:
        return "leonard"
    elif person_id == PENNY:
        return "penny"
    elif person_id == SHELDON:
        return "sheldon"
    elif person_id == HOWARD:
        return "howard"
    elif person_id == RAJ:
        return "raj"
    elif person_id == BERNADETTE:
        return "bernadette"
    elif person_id == AMY:
        return "amy"
    elif person_id == INVALID:
        return "invalid"
    else:
        raise ValueError('person_id not found')

def get_name_id(person_name):
    if person_name == "leonard":
        return LEONARD
    elif person_name == "penny":
        return PENNY
    elif person_name == "sheldon":
        return SHELDON
    elif person_name == "howard":
        return HOWARD
    elif person_name == "raj":
        return RAJ
    elif person_name == "bernadette":
        return BERNADETTE
    elif person_name == "amy":
        return AMY
    elif person_name == "invalid":
        return INVALID
    else:
        raise ValueError('person_name not found')

def get_face_in_box(this_frame, box):
    # print 'Get face in box ', box
    return this_frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2], :]


def get_pts_in_box(box):
    return (box[0], box[1]), (box[0] + box[2], box[1] + box[3])


def recognize_person(face_image):
    (label, score) = cnn_classifier.run_data(face_image)
    result = get_name_id(label)
    # print 'Face identified: ', label, score, result
    return result

def get_frame(frame_num):
    frame_filename = frames_folder + '/' + "%05d.jpg" % frame_num
    frame_img = cv2.imread(frame_filename)
    return frame_img

def round_it_up(float_box):
    round_box = []
    for this_point in float_box:
        round_box.append(int(round(this_point)))
    return tuple(round_box)

def track_faces(frame_limit=-1):
    global end_frame_num
    if frame_limit is not -1:
        end_frame_num = start_frame_num + frame_limit

    for this_face in faces:
        # print 'Tracker init - face: ', this_face
        mf_tracker = cv2.TrackerMedianFlow_create()
        face_tracker[this_face] = mf_tracker

    this_frame_num = start_frame_num
    # print box_in_frame.keys()
    # print box_in_frame.items()
    while not (this_frame_num in box_in_frame.keys() and this_frame_num < end_frame_num):
        this_frame_num += 1
    next_frame_num = this_frame_num + 1
    # while not (next_frame_num in box_in_frame.keys()):
    #     next_frame_num += 1

    while this_frame_num <= end_frame_num:
        this_frame_img = get_frame(this_frame_num)
        next_frame_img = get_frame(next_frame_num)
        # print 'Now: ', this_frame_num, 'Next: ', next_frame_num
        # print this_frame.shape
        if (frame_limit > 0 and this_frame_num >= frame_limit):
            break

        this_frame_boxes = box_in_frame[this_frame_num]
        for this_box in this_frame_boxes:
            if (this_frame_num, this_box) not in person_in_frame_box:
                this_face = get_face_in_box(this_frame_img, this_box)
                this_person = recognize_person(this_face)
            else:
                this_person = person_in_frame_box.get((this_frame_num, this_box))

            if this_person is not INVALID:
                # if face_tracker[this_person].algorithm.empty():
                face_tracker[this_person].init(this_frame_img, this_box)
                person_in_frame_box[(this_frame_num, this_box)] = this_person
                tracked_next_frame_box[(this_frame_num, this_person)] = this_box
                [success, next_box] = face_tracker[this_person].update(next_frame_img)
                if success:
                    next_box = round_it_up(next_box)
                    # next_face = get_face_in_box(next_frame_img, next_box)
                    # next_person = recognize_person(next_face)
                    # if this_person is not INVALID:
                    #     tracked_next_frame_box[(next_frame_num, next_person)] = next_box
                    #     person_in_frame_box[(next_frame_num, next_box)] = next_person
                    #     print 'Tracker - Frame: ', next_frame_num, '  Person: ', next_person
                    # else:
                    #     continue

                    # # No need to call face recognition every time.
                    # # Only if tracker fails
                    tracked_next_frame_box[(next_frame_num, this_person)] = next_box
                    person_in_frame_box[(next_frame_num, next_box)] = this_person
                else:
                    continue
            else:
                continue

        this_frame_num += 1
        while this_frame_num not in box_in_frame.keys() and this_frame_num < end_frame_num:
            this_frame_num += 1
        next_frame_num = this_frame_num + 1

        # while not (next_frame_num in box_in_frame.keys()):
        #     next_frame_num += 1

    print "Saving data"
    pickle.dump(person_in_frame_box, person_in_frame_box_fp)
    pickle.dump(tracked_next_frame_box, tracked_next_frame_box_fp)

    person_in_frame_box_fp.close()
    tracked_next_frame_box_fp.close()


def imwrite_output(frame_limit=-1):
    global end_frame_num
    if frame_limit is not -1:
        end_frame_num = start_frame_num + frame_limit

    for this_frame_num in range(start_frame_num, end_frame_num):
        print this_frame_num,
        this_frame_img = get_frame(this_frame_num)
        for this_person in faces:
            if (this_frame_num,this_person) in tracked_next_frame_box.keys():
                print this_person,
                this_box = tracked_next_frame_box.get((this_frame_num,this_person))
                [this_pt1, this_pt2] = get_pts_in_box(this_box)
                cv2.rectangle(this_frame_img, this_pt1, this_pt2, personColors[this_person - 1])
                cv2.putText(this_frame_img, get_name_string(this_person), this_pt1, font, fontScale, personColors[this_person - 1], lineType)

        frame_filename = output_frames_folder + '/' + "%05d.jpg" % this_frame_num
        cv2.imwrite(frame_filename, this_frame_img)
        print ''

def collect_boxes_in_frames():
    count = 0
    for box_file in listdir(boxes_folder):
        if box_file.endswith('.pick'):
            frame_num = int(box_file.split('.')[0])
            box_fp = open(boxes_folder + '/' + box_file, 'r')
            raw_boxes_in_this_frame = pickle.load(box_fp)
            boxes_in_this_frame = []
            for raw_box in raw_boxes_in_this_frame:
                this_box = []
                for raw_point in raw_box:
                    this_point = int(round(raw_point))
                    this_box.append(this_point)
                if len(this_box) == 4:
                    boxes_in_this_frame.append(tuple(this_box))
                else:
                    print 'Frame: ', frame_num, this_box, 'This does not seem like a valid box!'
            print 'Frame: ', frame_num, ' Boxes: ', boxes_in_this_frame
            box_in_frame[frame_num] = tuple(boxes_in_this_frame)
            count += 1
            # print count
    pickle.dump(box_in_frame, box_in_frame_fp)
    box_in_frame_fp.close()

if __name__ == '__main__':
    # collect_boxes_in_frames()
    track_faces(500)
    imwrite_output(500)

