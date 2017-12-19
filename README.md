# AdaptiveFaceTracking

Computer vision project attempting to discover and track faces in unconstrained video.
As of right now face detection is the limiting facot 

###Training Tf Models

```
cd tf/tensorflow/
source settings.sh
./run.sh
```
Check run.sh script for settings used.
  
###Running Face Tracking
Make sure to place a relevant BigBangTheory clip in the root repo directory in .mp4 format.
Set the video_path variable in detect_and_label_video.py
```
video_path = "./TBBT_S10E16.mp4"
```
Then run the script
```
./run_video.sh
```
