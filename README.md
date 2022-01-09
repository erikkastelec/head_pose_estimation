# head_pose_estimation

Repository contains implementation of Dlib face detector as well as Mediapipe Facemesh detector for head pose estimation. 


## Installation

Run `pip install -r requirements.txt` in the root directory of the project.

## Mediapipe Facemesh detector
Code for the detector is located inside mediapipe_facemesh_head_pose_estimator.py

Runs real time head pose estimation on the webcam input and shows the output in a new window.

`python mediapipe_facemesh_head_pose_estimator.py`

Runs head pose estimation on the whole video and saves it to out_path

`python mediapipe_facemesh_head_pose_estimator.py --video True --video_path "video_path" --out_path "out_path`

Returns pitch, yaw roll for the provided image

`python mediapipe_facemesh_head_pose_estimator.py --image True --image_path "image_path" `

## Dlib detector
Code for the detector is located inside dlib_head_pose_estimator.py 

Runs real time head pose estimation on the webcam input and shows the output in a new window. <br />
Scale percent option (--scale_percent x) scales the image to x% of initial size.

`python dlib_head_pose_estimator.py  --scale_percent 30`

Runs head pose estimation on the whole video and saves it to out_path. <br />
Scale percent option (--scale_percent x) scales the image to x% of initial size.

`python dlib_head_pose_estimator.py --video True --video_path "video_path" --out_path "out_path --scale_percent 30`

Returns pitch, yaw roll for the provided image. 

`python dlib_head_pose_estimator.py --image True --image_path "image_path" `
