import cv2
vidcap = cv2.VideoCapture('./video/TBBT_S10E16.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  print('Read frame: ', count)
  cv2.imwrite("./frames/%05d.jpg" % count, image)     # save frame as JPEG file
  count += 1
