import pickle
import cv2
import glob
from os import listdir
from os.path import isfile, join
import numpy as np

home_folder = '/home/gavincangan/computerVision/AdaptiveFaceTracking/'

box_in_frame_file = home_folder + 'box_in_frame.pick'
person_in_frame_box_file = home_folder + 'person_in_frame_box.pick'
tracked_next_frame_box_file = home_folder + 'tracked_next_frame_box.pick'

box_in_frame_fp = open(box_in_frame_file, 'r+b')
person_in_frame_box_fp = open(person_in_frame_box_file, 'r+b')
tracked_next_frame_box_fp = open(tracked_next_frame_box_file, 'r+b')

box_in_frame = pickle.load(box_in_frame_fp)
person_in_frame_box = dict() #pickle.load(person_in_frame_box_fp)
tracked_next_frame_box = dict() #pickle.load(tracked_next_frame_box_fp)

frames_folder = home_folder + 'frames'
boxes_folder = home_folder + 'pickled_boxes'

INVALID = -1
LEONARD = 1
PENNY = 2
SHELDON = 3

faces = [SHELDON, LEONARD, PENNY]
face_tracker = dict()

start_frame_num = 80
end_frame_num = 24499

def get_face_in_box(this_frame, box):
    print 'Get face in box ', box
    return this_frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2], :]

def recognize_person(face_image):
    print 'Face identified: ', SHELDON
    return SHELDON

def get_frame(frame_num):
    frame_filename = frames_folder + '/' + "%05d.jpg" % frame_num
    frame_img = cv2.imread(frame_filename)
    return frame_img

def round_it_up(float_box):
    round_box = []
    for this_point in float_box:
        round_box.append(int(round(this_point)))
    return tuple(round_box)

def track_faces():
    for this_face in faces:
        print 'Tracker init - face: ', this_face
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
        print 'Now: ', this_frame_num, 'Next: ', next_frame_num
        # print this_frame.shape

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
                    next_face = get_face_in_box(next_frame_img, next_box)
                    next_person = recognize_person(next_face)
                    if this_person is not INVALID:
                        tracked_next_frame_box[(next_frame_num, next_person)] = next_box
                        person_in_frame_box[(next_frame_num, next_box)] = next_person
                        print 'Tracker - Frame: ', next_frame_num, '  Person: ', next_person
                    else:
                        continue
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

    pickle.dump(person_in_frame_box, person_in_frame_box_fp)
    pickle.dump(tracked_next_frame_box, tracked_next_frame_box_fp)

    person_in_frame_box_fp.close()
    tracked_next_frame_box_fp.close()


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
    track_faces()

