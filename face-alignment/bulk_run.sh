#! /bin/bash

echo "Make sure the --imdir points to the source image directory"
echo "Make sure the --dest points to the destination image directory"
echo "Running sheldon"
python ./bulk_align.py --shape-predictor shape_predictor_68_face_landmarks.dat --imdir ../raw_bbt/sheldon/ --dest ../labeled_bbt/sheldon/
echo "Running leonard"
python ./bulk_align.py --shape-predictor shape_predictor_68_face_landmarks.dat --imdir ../raw_bbt/leonard/ --dest ../labeled_bbt/leonard/
echo "Running penny"
python ./bulk_align.py --shape-predictor shape_predictor_68_face_landmarks.dat --imdir ../raw_bbt/penny/ --dest ../labeled_bbt/penny/
