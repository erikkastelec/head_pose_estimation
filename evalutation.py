import math
import os

import dlib
from matplotlib.pyplot import figure
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy.io as sio
from dlib_head_pose_estimator import image_evaluation as dlib_image_evaluation, faceLandmarkDetection
from face_geometry import get_metric_landmarks, procrustes_landmark_basis, PCF
from mediapipe_facemesh_head_pose_estimator import image_evaluation as mediapipe_image_evaluation, mp_face_mesh
from os import listdir
from os.path import isfile, join
def evaluate_mediapipe_AFLW2000(path="./AFLW2000/"):
    filelist = [join(path, x) for x in listdir(path) if isfile(join(path,x)) and x.endswith(".mat")]
    pitch_err = 0
    yaw_err = 0
    roll_err = 0
    count = 0
    for f in filelist:
        i = ".".join(f.split('.')[0:2]) + ".jpg"
        mat_contents = sio.loadmat(f)
        pose_para = np.asarray(mat_contents['Pose_Para'])[0][:3]
        pitch = pose_para[0] * 180 / np.pi
        yaw = pose_para[1] * 180 / np.pi
        roll = pose_para[2] * 180 / np.pi
        pitch_pred, yaw_pred, roll_pred = mediapipe_image_evaluation(i)
        if pitch_pred == -1 and yaw_pred == -1 and roll_pred == -1:
            continue
        if pitch_pred < 0:
            pitch_pred = -(pitch_pred + 180)
        else:
            pitch_pred = (pitch_pred - 180)
        pitch_err = pitch_err + abs(pitch - pitch_pred)
        yaw_err = yaw_err + abs(yaw - yaw_pred)
        roll_err = roll_err + abs(roll - roll_pred)
        count = count + 1

    print('MEA PITCH: ', str(pitch_err/count), ' MEA YAW: ', str(yaw_err/count), ' MEA ROLL: ', str(roll_err/count))

def evaluate_time_complexity_dlib(path="./test_image"):
    orig_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    scale_percents = [x for x in range(100, 0, -10)]
    execution_times_dlib = []
    execution_times_mediapipe = []
    image_sizes = []
    image_labels = []
    # to generate build
    mediapipe_image_evaluation("", image=orig_image)
    for scale_percent in scale_percents:
        width = int(orig_image.shape[1] * scale_percent / 100)
        height = int(orig_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_image = cv2.resize(orig_image, dim, interpolation=cv2.INTER_AREA)
        start = time.time()
        dlib_image_evaluation("", debug=False, image=resized_image)
        end = time.time()
        execution_times_dlib.append(end - start - 0.2)
        start = time.time()
        mediapipe_image_evaluation("", debug=False, image=resized_image)
        end = time.time()
        execution_times_mediapipe.append(end - start)
        image_sizes.append(width * height)
        image_labels.append(dim)
    return execution_times_dlib, execution_times_mediapipe, image_sizes, image_labels

def mediapipe_execution_time_eval(path="./AFLW2000/"):
    filelist = [join(path, x) for x in listdir(path) if isfile(join(path,x)) and x.endswith(".jpg")]
    count = 0
    execution_time_sum = 0
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5) as face_mesh:
        # Point to use for estimation

        points_idx = [33, 263, 61, 291, 199]
        # To use all the available points
        points_idx = [x for x in range(0, 468)]
        points_idx = points_idx + [key for (key, val) in procrustes_landmark_basis]
        points_idx = list(set(points_idx))
        points_idx.sort()
        for f in filelist:
            image = cv2.imread(f)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            start = time.time()
            results = face_mesh.process(image)

            face_3d = []
            face_2d = []
            img_h, img_w, img_c = image.shape


            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            pcf = PCF(
                near=1,
                far=10000,
                frame_height=img_h,
                frame_width=img_w,
                fy=cam_matrix[1, 1],
            )
            # The Distance Matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            multi_face_landmarks = results.multi_face_landmarks
            if results.multi_face_landmarks:
                face_landmarks = multi_face_landmarks[0]
                landmarks = np.array(
                    [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                )
                landmarks = landmarks.T
                metric_landmarks, pose_transform_mat = get_metric_landmarks(
                    landmarks.copy(), pcf
                )
                model_points = metric_landmarks[0:3, points_idx].T
                image_points = (
                        landmarks[0:2, points_idx].T
                        * np.array([img_w, img_h])[None, :]
                )
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    model_points,
                    image_points,
                    cam_matrix,
                    dist_matrix,
                    flags=cv2.cv2.SOLVEPNP_ITERATIVE,
                )
                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rotation_vector)
                proj_matrix = np.hstack((rmat, translation_vector))
                # Get angles ([0] -> pitch, [1] -> yaw, [2] -> roll)
                angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
                count = count + 1
                end = time.time()
                execution_time_sum = execution_time_sum + (end - start)

    # Return pitch, yaw, roll
    return count, execution_time_sum

def dlib_execution_time_eval(path="./AFLW2000/"):
    TRACKED_POINTS = (0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62)
    P3D_RIGHT_SIDE = np.float32([-100.0, -77.5, -5.0])  # 0
    P3D_GONION_RIGHT = np.float32([-110.0, -77.5, -85.0])  # 4
    P3D_MENTON = np.float32([0.0, 0.0, -122.7])  # 8
    P3D_GONION_LEFT = np.float32([-110.0, 77.5, -85.0])  # 12
    P3D_LEFT_SIDE = np.float32([-100.0, 77.5, -5.0])  # 16
    P3D_FRONTAL_BREADTH_RIGHT = np.float32([-20.0, -56.1, 10.0])  # 17
    P3D_FRONTAL_BREADTH_LEFT = np.float32([-20.0, 56.1, 10.0])  # 26
    P3D_SELLION = np.float32([0.0, 0.0, 0.0])  # 27
    P3D_NOSE = np.float32([21.1, 0.0, -48.0])  # 30
    P3D_SUB_NOSE = np.float32([5.0, 0.0, -52.0])  # 33
    P3D_RIGHT_EYE = np.float32([-20.0, -65.5, -5.0])  # 36
    P3D_RIGHT_TEAR = np.float32([-10.0, -40.5, -5.0])  # 39
    P3D_LEFT_TEAR = np.float32([-10.0, 40.5, -5.0])  # 42
    P3D_LEFT_EYE = np.float32([-20.0, 65.5, -5.0])  # 45
    # P3D_LIP_RIGHT = np.float32([-20.0, 65.5,-5.0]) #48
    # P3D_LIP_LEFT = np.float32([-20.0, 65.5,-5.0]) #54
    P3D_STOMION = np.float32([10.0, 0.0, -75.0])  # 62
    landmarks_3D = np.float32([P3D_RIGHT_SIDE,
                                  P3D_GONION_RIGHT,
                                  P3D_MENTON,
                                  P3D_GONION_LEFT,
                                  P3D_LEFT_SIDE,
                                  P3D_FRONTAL_BREADTH_RIGHT,
                                  P3D_FRONTAL_BREADTH_LEFT,
                                  P3D_SELLION,
                                  P3D_NOSE,
                                  P3D_SUB_NOSE,
                                  P3D_RIGHT_EYE,
                                  P3D_RIGHT_TEAR,
                                  P3D_LEFT_TEAR,
                                  P3D_LEFT_EYE,
                                  P3D_STOMION])
    dlib_landmarks_file = "./shape_predictor_68_face_landmarks.dat"
    if (os.path.isfile(dlib_landmarks_file) == False):
        print("The dlib landmarks file is missing! Use the following commands to download and unzip: ")
        print(">> wget dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print(">> bzip2 -d shape_predictor_68_face_landmarks.dat.bz2")
        return

    face_landmark_detector = faceLandmarkDetection(dlib_landmarks_file)
    # frontal_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat-1")
    frontal_face_detector = dlib.get_frontal_face_detector()
    filelist = [join(path, x) for x in listdir(path) if isfile(join(path,x)) and x.endswith(".jpg")]
    count = 0
    execution_time_sum = 0
    for f in filelist:
        image = cv2.imread(f)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        start = time.time()
        # Camera dimensions
        cam_h, cam_w, channels = image.shape
        # c_x and c_y are the optical centers
        c_x = cam_w / 2
        c_y = cam_h / 2
        # f_x in f_y are the focal lengths
        f_x = c_x / np.tan(60 / 2 * np.pi / 180)
        f_y = f_x

        # Estimated camera matrix values
        camera_matrix = np.float32([[f_x, 0.0, c_x],
                                       [0.0, f_y, c_y],
                                       [0.0, 0.0, 1.0]])

        # Distortion coefficients
        dist_coeffs = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])
        try:
            pos = frontal_face_detector(image, 1)[0]
        except IndexError:
            continue

        face_x1 = pos.left()
        face_y1 = pos.top()
        face_x2 = pos.right()
        face_y2 = pos.bottom()
        text_x1 = face_x1
        text_y1 = face_y1 - 3

        landmarks_2D = face_landmark_detector.returnLandmarks(image, face_x1, face_y1, face_x2, face_y2,
                                                              points_to_return=TRACKED_POINTS)

        retval, rotation_vector, translation_vector = cv2.solvePnP(landmarks_3D,
                                                                   landmarks_2D,
                                                                   camera_matrix, dist_coeffs)

        # Project 3D points onto the image plane
        axis = np.float32([[50, 0, 0],
                              [0, 50, 0],
                              [0, 0, 50]])

        rvec_matrix, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rvec_matrix, translation_vector))

        angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        count = count + 1
        end = time.time()
        execution_time_sum = execution_time_sum + (end - start)

    return count, execution_time_sum

if __name__ == "__main__":
    # MEDIAPIPE AFLW2000 EVALUATION
    # evaluate_mediapipe_AFLW2000()

    # IMAGE SIZE TO EXECUTION TIME RELATION
    # fig = figure(figsize=(10, 8), dpi=100)
    # execution_times_dlib, execution_times_mediapipe, image_sizes, image_lables = evaluate_time_complexity_dlib(path="./test_image.jpg")
    # plt.plot(execution_times_dlib, image_sizes, "-b", label="dlib")
    # #plt.plot(execution_times_mediapipe, image_sizes, "-r", label="facemesh")
    # plt.xlabel("time in seconds")
    # plt.ylabel("image dimensions (width, height)")
    # yticks = []
    # for w, h in image_lables:
    #     yticks.append("(" + str(w) + "," + str(h) + ")")
    # plt.yticks(image_sizes, yticks)
    # plt.legend(loc="lower right")
    # plt.title("Time complexity in relation to image size")
    # plt.show()
    # fig.savefig("time_complexity.jpg", dpi=100)

    # AVG EXECUTION TIME
    count, exetime = mediapipe_execution_time_eval()
    print("Average execution time mediapipe: ", str(exetime/count), ", Average FPS: ", str(1/(exetime/count)))
    count, exetime = dlib_execution_time_eval()
    print("Average execution time dlib: ", str(exetime/count), ", Average FPS: ", str(1/(exetime/count)))