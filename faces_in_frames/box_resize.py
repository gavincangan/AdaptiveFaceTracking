import sys
import pickle

input_box_file = sys.argv[1]
output_box_file = '../boxes_pickled/' + input_box_file

im_width_act = 1280
im_width_resized = 800

im_resize_ratio = float(im_width_act)/float(im_width_resized)

box_in_fp = open(input_box_file)
boxes_in = pickle.load(box_in_fp)

boxes = []

for box_in in boxes_in:
    this_box = []
    for pt in box_in:
        this_box.append(float(pt) * im_resize_ratio)
    boxes.append(this_box)

box_out_fp = open('output_box_file', 'w')
pickle.dump(boxes, box_out_fp)

print input_box_file, 'saved'