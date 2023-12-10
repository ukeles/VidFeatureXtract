#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Some helper functions for face and bodypart detection pipelines. 

"""

import cv2
import numpy as np

# ----- utils for face detection -----
def faces2dets(faces):
    """
    Convert face detections into a formatted array.

    This function reformats detected faces from an input list into a single array 
    for each detected face, where each array contains the bounding box coordinates,
    detection score, and key points.

    Parameters
    ----------
    faces : list
        List of detected face objects, where each object has attributes:
        - bbox: Bounding box coordinates [x1, y1, x2, y2].
        - det_score: Detection score.
        - kps: 2D key points array.

    Returns
    -------
    numpy.ndarray
        Array of reformatted detections. Each row represents a detected face with
        bounding box coordinates, detection score, and flattened key points.

    """
    # Check if there are any detected faces.
    if len(faces) < 1:
        # No face detected for the image/frame.
        return []
    else:
        dets = []
        for fc_ii in faces:
            # Reformat detection for the current face into a single vector.
            dets_this = [ *fc_ii.bbox, fc_ii.det_score, *fc_ii.kps.flatten() ]
            dets.append(dets_this)
    return np.asarray(dets)


def get_faceareas_simple(this_frame_results, frame_height, frame_width,
                         detection_thrs=0.5, sort_by_area=False,
                         return_corners=False, return_landmarks_full=False):
    """
    Process face detection results and compute a mask indicating face areas.

    Parameters
    ----------
    this_frame_results : numpy.ndarray
        2D array where each row represents a face detection. Columns are:
        [x1, y1, x2, y2, detection_score, ...landmarks...].
    frame_height : int
        Height of the frame.
    frame_width : int
        Width of the frame.
    detection_thrs : float, optional
        Detection threshold to consider face as valid. Default is 0.5.
    sort_by_area : bool, optional
        Sort detected faces based on their area instead of detection score. Default is False.
    return_corners : bool, optional
        Return the corners of the detected faces. Default is False.
    return_landmarks_full : bool, optional
        Return all landmarks instead of just the eyes, nose, and mouth center. Default is False.

    Returns
    -------
    face_areas : numpy.ndarray
        2D array of shape (frame_height, frame_width) indicating detected face areas.
    landmarks_all : list of numpy.ndarray
        List of landmarks for each detected face.
    corners : list of list (optional)
        List of corners [x1, y1, x2, y2] for each detected face.
    """

    # Compute areas for sorting (if required).
    if sort_by_area:
        areas = (this_frame_results[:, 3] - this_frame_results[:, 1]) * (this_frame_results[:, 2] - this_frame_results[:, 0])
        this_frame_results = this_frame_results[np.argsort(areas)[::-1]]

    face_areas = np.zeros((frame_height, frame_width))

    landmarks_all = []
    corners = [] if return_corners else None

    face_num = 1
    for hii in this_frame_results:
        if hii[4] > detection_thrs:
            # Bounding the coordinates within the frame dimensions.
            # this line ensures that all x-coordinates are within [0, frame_width - 1], 
            # all y-coordinates are within [0, frame_height - 1], and the detection score is above 0--trivial.
            b = np.clip(hii.astype(int), 0, 
                        [frame_width-1, frame_height-1, frame_width-1, frame_height-1, np.inf] +\
                        [frame_width-1, frame_height-1] * 5).astype(int)

            face_areas[b[1]:b[3]+1, b[0]:b[2]+1] = face_num
            face_num += 1
            
            landmarks = b[5:].reshape(-1, 2)
            #from viewers perspective: b5,b6: left eye; b7,b8: right eye;  
            # b9,b10: nose; b11,b12: left-side mouth; b13,b14: right-side mouth
            
            mouth_center = landmarks[3:5, :].mean(axis=0).round().astype(int)
            
            if return_landmarks_full:
                landmarks_all.append(landmarks)
            else:
                landmarks_all.append(np.vstack((landmarks[:-2, :], mouth_center)))

            if return_corners:
                corners.append(b[:4])

    if return_corners:
        return face_areas, landmarks_all, corners

    return face_areas, landmarks_all


def get_faceareas(this_frame_results, frame_height, frame_width, detection_thrs=0.5,
                  return_mask=True, return_box=False):
    """
    Compute binary masks and/or bounding boxes for the detected face regions, 
    eye regions, and mouth regions in the frame.
    
    Parameters
    ----------
    this_frame_results : array-like
        2D array where each row contains bounding box coordinates, detection score, and facial landmarks.
        Expected columns are:
        x1, y1, x2, y2 (bounding box), detection score, and x, y coordinates for five facial landmarks.
    frame_height : int
        Height of the frame.
    frame_width : int
        Width of the frame.
    detection_thrs : float, optional
        Detection threshold for face detection. Faces with a score below this value will be ignored.
        Default is 0.5.
    return_mask : bool, optional
        If True, returns binary masks for detected regions. Default is True.
    return_box : bool, optional
        If True, returns bounding boxes for detected regions. Default is False.
        
    Returns
    -------
    face_areas or face_boxes : np.ndarray or list
        Binary mask or bounding boxes for the face region.
    face_areas_eyes or face_boxes_eyes : np.ndarray or list
        Binary mask or bounding boxes for the eye region.
    face_areas_mouth or face_boxes_mouth : np.ndarray or list
        Binary mask or bounding boxes for the mouth region.
    """
    
    if return_mask:
        # Initializing binary masks
        face_areas = np.zeros((frame_height, frame_width), dtype=bool)
        face_areas_eyes = np.zeros((frame_height, frame_width), dtype=bool)
        face_areas_mouth = np.zeros((frame_height, frame_width), dtype=bool)

    if return_box:
        face_boxes = []
        face_boxes_eyes = []
        face_boxes_mouth = []        

    for hii in this_frame_results:
        if hii[4] > detection_thrs:
            # Bounding the coordinates within the frame dimensions.
            # this line ensures that all x-coordinates are within [0, frame_width - 1], 
            # all y-coordinates are within [0, frame_height - 1], and the detection score is above 0--trivial.
            b = np.clip(hii.astype(int), 0, 
                        [frame_width-1, frame_height-1, frame_width-1, frame_height-1, np.inf] +\
                        [frame_width-1, frame_height-1] * 5).astype(int)
            
            if return_mask:
                # Updating the face region mask
                face_areas[b[1]:b[3]+1, b[0]:b[2]+1] = True

            if return_box:
                face_boxes.append([b[1], b[3], b[0], b[2]])
               
            landmarks = b[5:].reshape(-1, 2)

            # Computing a combined bounding box using both the detected face and landmark positions
            box2 = [np.min(landmarks[:, 0]), np.min(landmarks[:, 1]), np.max(landmarks[:, 0]), np.max(landmarks[:, 1])]
            med_box = np.vstack((b[:4], box2)).mean(0).astype(int).reshape(-1, 2)

            # Control for out of frame detections.
            med_box = np.clip(med_box, [0, 0], [frame_width - 1, frame_height - 1])
            med_box = med_box.flatten()

            # Computing regions for eyes and mouth based on the combined bounding box and nose position
            box_11 = [np.min([med_box[0], med_box[2], landmarks[2, 0]]), np.min([med_box[1], landmarks[2, 1]]),
                      np.max([med_box[0], med_box[2], landmarks[2, 0]]), np.max([med_box[1], landmarks[2, 1]])]
            
            box_22 = [np.min([med_box[0], med_box[2], landmarks[2, 0]]), np.min([med_box[3], landmarks[2, 1]]),
                      np.max([med_box[0], med_box[2], landmarks[2, 0]]), np.max([med_box[3], landmarks[2, 1]])]


            if return_mask:
                face_areas_eyes[box_11[1]:box_11[3]+1, box_11[0]:box_11[2]+1] = True
                face_areas_mouth[box_22[1]:box_22[3]+1, box_22[0]:box_22[2]+1] = True

            if return_box:
                face_boxes_eyes.append([box_11[1], box_11[3], box_11[0], box_11[2]])
                face_boxes_mouth.append([box_22[1], box_22[3], box_22[0], box_22[2]])

    if return_mask and not return_box:
        return face_areas, face_areas_eyes, face_areas_mouth
    elif not return_mask and return_box:
        return face_boxes, face_boxes_eyes, face_boxes_mouth
    else:
        return face_areas, face_areas_eyes, face_areas_mouth, face_boxes, face_boxes_eyes, face_boxes_mouth


def visualize_facedetections(img, face_boxes, face_boxes_eyes, face_boxes_mouth):
    """
    Helper function to visualize face detections on a single frame.

    Parameters
    ----------
    img : numpy.ndarray
        The image on which to draw the visualizations.
    face_boxes : list
        A list of face bounding boxes.
    face_boxes_eyes : list
        A list of eye region bounding boxes.
    face_boxes_mouth : list
        A list of mouth region bounding boxes.
    
    Returns
    -------
    None
    """
    for face_cnt in range(len(face_boxes)):
        draw_bounding_box(img, face_boxes[face_cnt], color=(255, 255, 255))
        draw_bounding_box(img, face_boxes_eyes[face_cnt], color=(0, 140, 255))
        draw_bounding_box(img, face_boxes_mouth[face_cnt], color=(208, 224, 64))


def draw_bounding_box(img, box, color):
    """
    Draws a bounding box on an image.

    Parameters
    ----------
    img : numpy.ndarray
        The image on which to draw the box.
    box : list
        The bounding box with coordinates [y_min, y_max, x_min, x_max].
    color : tuple
        The color of the box (B, G, R).
    
    Returns
    -------
    None
    """
    cv2.rectangle(img, (box[2], box[0]), (box[3], box[1]), color, thickness=2)


    

'''
NOTE:
YOLOv8_face is taken and modified from: 
    https://github.com/hpc203/yolov8-face-landmarks-opencv-dnn
'''
import math
class YOLOv8_face:
    def __init__(self, path, conf_thres=0.2, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.class_names = ['face']
        self.num_classes = len(self.class_names)
        # Initialize model
        self.net = cv2.dnn.readNet(path)
        self.input_height = 640
        self.input_width = 640
        self.reg_max = 16

        self.project = np.arange(self.reg_max)
        self.strides = (8, 16, 32)
        self.feats_hw = [(math.ceil(self.input_height / self.strides[i]), math.ceil(self.input_width / self.strides[i])) for i in range(len(self.strides))]
        self.anchors = self.make_anchors(self.feats_hw)

    def make_anchors(self, feats_hw, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points = {}
        for i, stride in enumerate(self.strides):
            h,w = feats_hw[i]
            x = np.arange(0, w) + grid_cell_offset  # shift x
            y = np.arange(0, h) + grid_cell_offset  # shift y
            sx, sy = np.meshgrid(x, y)
            # sy, sx = np.meshgrid(y, x)
            anchor_points[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2)
        return anchor_points

    def softmax(self, x, axis=1):
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s
    
    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_width, self.input_height
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_height, int(self.input_width / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_width - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_width - neww - left, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))  # add border
            else:
                newh, neww = int(self.input_height * hw_scale), self.input_width
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_height - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_height - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))
        else:
            img = cv2.resize(srcimg, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def detect(self, srcimg):
        input_img, newh, neww, padh, padw = self.resize_image(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB))
        scale_h, scale_w = srcimg.shape[0]/newh, srcimg.shape[1]/neww
        input_img = input_img.astype(np.float32) / 255.0

        blob = cv2.dnn.blobFromImage(input_img)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        det_bboxes, det_conf, det_classid, landmarks = self.post_process(outputs, scale_h, scale_w, padh, padw)
        
        landmarks_inds = np.asarray([0, 1, 3, 4, 6, 7, 9, 10, 12, 13])
        landmarks = landmarks[:,landmarks_inds]
        return det_bboxes, det_conf[:,None], det_classid, landmarks

    def post_process(self, preds, scale_h, scale_w, padh, padw):
        bboxes, scores, landmarks = [], [], []
        for i, pred in enumerate(preds):
            stride = int(self.input_height/pred.shape[2])
            pred = pred.transpose((0, 2, 3, 1))
            
            box = pred[..., :self.reg_max * 4]
            cls = 1 / (1 + np.exp(-pred[..., self.reg_max * 4:-15])).reshape((-1,1))
            kpts = pred[..., -15:].reshape((-1,15)) ### x1,y1,score1, ..., x5,y5,score5

            # tmp = box.reshape(self.feats_hw[i][0], self.feats_hw[i][1], 4, self.reg_max)
            tmp = box.reshape(-1, 4, self.reg_max)
            bbox_pred = self.softmax(tmp, axis=-1)
            bbox_pred = np.dot(bbox_pred, self.project).reshape((-1,4))

            bbox = self.distance2bbox(self.anchors[stride], bbox_pred, max_shape=(self.input_height, self.input_width)) * stride
            kpts[:, 0::3] = (kpts[:, 0::3] * 2.0 + (self.anchors[stride][:, 0].reshape((-1,1)) - 0.5)) * stride
            kpts[:, 1::3] = (kpts[:, 1::3] * 2.0 + (self.anchors[stride][:, 1].reshape((-1,1)) - 0.5)) * stride
            kpts[:, 2::3] = 1 / (1+np.exp(-kpts[:, 2::3]))

            bbox -= np.array([[padw, padh, padw, padh]]) 
            bbox *= np.array([[scale_w, scale_h, scale_w, scale_h]])
            kpts -= np.tile(np.array([padw, padh, 0]), 5).reshape((1,15))
            kpts *= np.tile(np.array([scale_w, scale_h, 1]), 5).reshape((1,15))

            bboxes.append(bbox)
            scores.append(cls)
            landmarks.append(kpts)

        bboxes = np.concatenate(bboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        landmarks = np.concatenate(landmarks, axis=0)
    
        bboxes_wh = bboxes.copy()
        
        bboxes_wh[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]  ####xywh
        classIds = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)  ####max_class_confidence
        
        mask = confidences>self.conf_threshold
        bboxes_xy = bboxes[mask]
        bboxes_wh = bboxes_wh[mask]
        confidences = confidences[mask]
        classIds = classIds[mask]
        landmarks = landmarks[mask]
        
        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.conf_threshold,
                                   self.iou_threshold).flatten()
        if len(indices) > 0:
            mlvl_bboxes = bboxes_xy[indices]
            # mlvl_bboxes = bboxes_wh[indices]
            confidences = confidences[indices]
            classIds = classIds[indices]
            landmarks = landmarks[indices]
            return mlvl_bboxes, confidences, classIds, landmarks
        else:
            print('nothing detect')
            return np.array([]), np.array([]), np.array([]), np.array([])

    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)
    
    def draw_detections(self, image, boxes, scores, kpts):
        for box, score, kp in zip(boxes, scores, kpts):
            x1, y1, x2, y2 = box.astype(int)
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
            # cv2.putText(image, "face:"+str(round(score,2)), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
            for i in range(5):
                cv2.circle(image, (int(kp[i * 2]), int(kp[i * 2 + 1])), 2, (0, 255, 0), thickness=-1)
        return image



# -------- for bodyparts obtained from densepose --------
from ..utils.io_helpers import str2array_decoder

import base64
from io import BytesIO
from PIL import Image


def densepose_reencoder(s_in):
    """
    Re-encodes the DensePose output, focusing on the body part index part of the IUV matrix.

    Parameters
    ----------
    s_in : str
        The base64 encoded string of the DensePose output.

    Returns
    -------
    str
        The re-encoded base64 string, now containing only the body part index 
        part of the IUV matrix.
    """
    # Decode the input string to retrieve the IUV matrix image
    fstream = BytesIO(base64.decodebytes(s_in.encode()))
    
    # Extract only the first channel (body part index) from the IUV matrix
    im_out = Image.open(fstream).getchannel(0)
    # im_out = Image.open(fstream).split()[0] # option 3. 
    # options 2 slightly slower
    # data_iuv = np.asarray(Image.open(fstream))
    # im_out = Image.fromarray(data_iuv[...,0])
    
    # Re-encode the extracted channel to a compressed form
    fstream = BytesIO()
    im_out.save(fstream, format="png", optimize=True)
    s_out = base64.encodebytes(fstream.getvalue()).decode()
    
    return s_out


def get_boundingbox_dense(arr_in):
    """
    Calculate the bounding box coordinates for the non-zero regions in a 2D array.

    Parameters
    ----------
    arr_ : np.ndarray
        A 2D numpy array where the non-zero elements represent the object of interest.

    Returns
    -------
    tuple
        The coordinates of the bounding box as (xmin, ymin, xmax, ymax), corresponding
        to the left, top, right, and bottom edges of the box.
    """
    # Identify rows and columns that contain non-zero elements
    rows = np.any(arr_in, axis=1)
    cols = np.any(arr_in, axis=0)
    # Find the indices of the first and last non-zero rows and columns
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # Return the coordinates as (xmin, ymin, xmax, ymax), i.e., [left, top, right, bottom]
    return cmin, rmin, cmax, rmax


def densepose_results2bodypart_layers(result_dpose, img_shape, pred_score_thrs=0.8, keep_nperson=5000):
    """
    Convert DensePose results to body part layers and extract head bounding boxes.

    Parameters
    ----------
    result_dpose : dict
        The DensePose result containing 'dp_scores', 'dp_boxes_xyxy', and 'dp_parts_str'.
    img_shape : tuple
        The shape of the image (height, width) on which DensePose was applied.
    pred_score_thrs : float, optional
        The threshold for considering a prediction. Default is 0.8.
    keep_nperson : int, optional
        The maximum number of persons to process. Default is 5000.

    Returns
    -------
    np.ndarray or None
        A 2D array representing body part layers if any person is detected, None otherwise.
    np.ndarray or None
        An array containing the bounding boxes for heads if detected, None otherwise.
    """
    
    # Initialize layers for body parts and a temporary placeholder
    bodyparts_layer = np.zeros(img_shape[:2], dtype=np.uint8)
    bodyparts_dum = np.zeros(img_shape[:2], dtype=np.uint8)

    # List to keep track of head bounding boxes
    keep_head_box = []

    # Iterate through each instance in the DensePose results
    for instance_id, pred_score in enumerate(result_dpose["dp_scores"]):
        # Check if the prediction score meets the threshold and within person limit
        if pred_score >= pred_score_thrs and instance_id < keep_nperson:
            pred_box = result_dpose['dp_boxes_xyxy'][instance_id]
            result_encoded = result_dpose['dp_parts_str'][instance_id]
            
            # Decode the body parts
            bodyparts_instance = str2array_decoder(result_encoded)
            
            # Reset the temporary layer and place the current instance's body parts
            bodyparts_dum[:] = 0
            y_offset, x_offset = pred_box[1], pred_box[0]
            bodyparts_dum[y_offset:y_offset + bodyparts_instance.shape[0],
                          x_offset:x_offset + bodyparts_instance.shape[1]] = bodyparts_instance
            
            # Indices for the head
            head_indx_1, head_indx_2 = 23, 24
            head_inds = np.logical_or(bodyparts_dum == head_indx_1, bodyparts_dum == head_indx_2)
            
            # Extract bounding box for the head if present
            if np.any(head_inds):
                x_min, y_min, x_max, y_max = get_boundingbox_dense(head_inds)
                keep_head_box.append([x_min, y_min, x_max, y_max, pred_score])
            
            # Update the bodyparts layer with the current instance
            bodyparts_layer[bodyparts_dum > 0] = bodyparts_dum[bodyparts_dum > 0]

    # Check for the presence of detected persons
    if np.any(bodyparts_layer):
        return bodyparts_layer, np.asarray(keep_head_box)
    else: # No person detected with sufficient prediction score
        return None, None


def densepose_facehand_layers(multiarea_layer):
    """
    Segregates body parts into head, hands, and other body parts based on DensePose indices.

    Parameters
    ----------
    multiarea_layer : np.ndarray
        A 2D numpy array where each element corresponds to a DensePose body part index.

    Returns
    -------
    np.ndarray
        A 2D numpy array with body parts segregated into head (value 3), hands (value 2), 
        and other body parts (value 1).
    """
    # Copy the input array to preserve the original data
    bodypart_inds = multiarea_layer.copy()

    # Initialize an array for the reduced body parts
    bodyparts_reduced = np.zeros_like(bodypart_inds)
    
    # Define indices for head and hands
    head_indx_1, head_indx_2 = 23, 24
    hand_indx_1, hand_indx_2 = 3, 4

    # Identify head and hand regions
    head_inds = np.logical_or(bodypart_inds == head_indx_1, bodypart_inds == head_indx_2)
    hand_inds = np.logical_or(bodypart_inds == hand_indx_1, bodypart_inds == hand_indx_2)
    
    # Assign unique values for head, hands, and other body parts
    bodyparts_reduced[head_inds] = 3  # Assign value 3 to head
    bodyparts_reduced[hand_inds] = 2  # Assign value 2 to hands
    
    # Exclude head and hand indices from the original indices
    bodypart_inds[head_inds | hand_inds] = 0
    # Assign value 1 to other body parts
    bodyparts_reduced[bodypart_inds > 0] = 1

    return bodyparts_reduced


def densepose_get_indvheads(result_dpose, img_shape, pred_score_thrs=0.8, keep_nperson=5000):
    """
    Extracts individual head regions and their bounding boxes from DensePose results.

    Parameters
    ----------
    result_dpose : dict
        The DensePose result containing 'dp_scores', 'dp_boxes_xyxy', and 'dp_parts_str'.
    img_shape : tuple
        The shape of the image (height, width) on which DensePose was applied.
    pred_score_thrs : float, optional
        The threshold for considering a prediction as valid. Default is 0.8.
    keep_nperson : int, optional
        The maximum number of persons to process. Default is 5000.

    Returns
    -------
    np.ndarray or None
        A 2D array with uniquely numbered head regions if detected, None otherwise.
    np.ndarray or None
        An array containing the bounding boxes for detected head regions, None otherwise.
    """
    
    # Initialize layers for body parts and a placeholder for head detections
    bodyparts_layer = np.zeros(img_shape[:2], dtype=np.uint8)
    bodyparts_dum = np.zeros(img_shape[:2], dtype=np.uint8)
    keep_head_box = []
    head_num = 1

    # Process each instance in the DensePose results
    for instance_id, pred_score in enumerate(result_dpose["dp_scores"]):
        if pred_score >= pred_score_thrs and instance_id < keep_nperson:
            # Extract the bounding box and decoded body parts
            pred_box = result_dpose['dp_boxes_xyxy'][instance_id]
            bodyparts_instance = str2array_decoder(result_dpose['dp_parts_str'][instance_id])
            
            # Reset and fill the temporary layer with the current instance's body parts
            bodyparts_dum[:] = 0
            y_offset, x_offset = pred_box[1], pred_box[0]
            bodyparts_dum[y_offset:y_offset + bodyparts_instance.shape[0],
                          x_offset:x_offset + bodyparts_instance.shape[1]] = bodyparts_instance
            
            # Detect head regions
            head_indx_1, head_indx_2 = 23, 24
            head_inds = np.logical_or(bodyparts_dum == head_indx_1, bodyparts_dum == head_indx_2)
            
            # Extract bounding box for the head region
            if np.any(head_inds):
                x_min, y_min, x_max, y_max = get_boundingbox_dense(head_inds)
                keep_head_box.append([x_min, y_min, x_max, y_max, pred_score])
            
            # Update the bodyparts layer with uniquely numbered head regions
            bodyparts_layer[head_inds] = head_num
            head_num += 1

    # Check if any head regions were detected
    if np.any(bodyparts_layer):
        return bodyparts_layer, np.asarray(keep_head_box)
    else: # No head region detected
        return None, None




def apply_colormask(image_cv, mask, color_rgb, alpha=0.5):
    """
    Overlays a colored mask onto an image with specified transparency.

    Parameters
    ----------
    image_cv : np.ndarray
        The original image onto which the mask will be applied.
    mask : np.ndarray
        A binary mask where 1 indicates the regions to be colored.
    color_rgb : tuple
        The color for the mask. Specified as (R, G, B).
    alpha : float
        The transparency level for the overlay. Default is 0.5.

    Returns
    -------
    np.ndarray
        The image with the overlay applied.
    """
    # Invert the color order to BGR format
    color_bgr = color_rgb[::-1]

    # Create a 3-channel colored mask based on the input mask
    colored_mask = np.dstack([mask] * 3)

    # Apply the color to the mask
    masked = np.ma.MaskedArray(image_cv, mask=colored_mask, fill_value=color_bgr)
    image_overlay = masked.filled()

    # Blend the original image with the overlay
    image_combined = cv2.addWeighted(image_cv, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


# Define colors for each layer value
bparts_colors = {
    0: None,         # Transparent for 0
    1: (165,0,255), # 1 for non-head/hand body parts | purple
    2: (0,255,165), # 2 for hands | green
    3: (255,165,0)  # 3 for heads | orange
    }

def visualize_bpartdetections(img, detection_mask):
    """
    Visualizes body part detections on an image using colored overlays.

    Parameters
    ----------
    img : np.ndarray
        The original image onto which the overlays will be applied.
    detection_mask : np.ndarray
        A 2D array where each value corresponds to a different body part.

    Returns
    -------
    np.ndarray
        The image with body part detection overlays applied.
    """
    # Map each value in detection_mask to its corresponding color
    for value, color in bparts_colors.items():
        if value:  # Skip 0 as it's meant to be transparent
            mask_i = detection_mask == value
            img = apply_colormask(img, mask_i, color_rgb=color, alpha=0.5)

    return img

