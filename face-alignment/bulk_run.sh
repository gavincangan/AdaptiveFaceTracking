#! /bin/bash

echo "Make sure the --imdir points to the source image directory"
python ./bulk_align.py --shape-predictor shape_predictor_68_face_landmarks.dat --imdir ../images/
