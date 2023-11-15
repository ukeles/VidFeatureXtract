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


def finalize_and_save(output_vid_file, outfile_pkl, feats_data, outfile_arr, feats_arr):
    """
    Finalizes video writing if applicable and saves feature data to files.

    Parameters
    ----------
    output_vid_file : cv2.VideoWriter
        The VideoWriter object used for saving the visualization video.
    outfile_pkl : str
        File path for the pickle file to save feature data.
    feats_data : dict
        The extracted feature data.
    outfile_arr : str
        File path for the numpy array to save aggregated feature data.
    feats_arr : numpy.ndarray
        The aggregated feature data for each frame.

    Returns
    -------
    None
    """
    if output_vid_file:
        output_vid_file.release()

    import pickle
    with open(outfile_pkl, 'wb') as pfile:
        pickle.dump(feats_data, pfile)
    
    np.save(outfile_arr, feats_arr)
    
    

#%% for bodyparts obtained from densepose

import base64
from io import BytesIO
from PIL import Image


def radmask(center_xy,radius,array_shape):
    
    if np.array(center_xy).shape != (2,): 
        raise SystemExit('Problem in the input of radMask')
    
    col_val = center_xy[0] 
    row_val = center_xy[1] 
    n_rows,n_cols = array_shape
    rows,cols = np.ogrid[-row_val:n_rows-row_val,-col_val:n_cols-col_val]
    mask = cols*cols + rows*rows <= radius*radius
    return mask    



def decode_png_data_original(shape, s):
    """
    From FaceBook's Detectron2/DensePose.
    Decode array data from a string that contains PNG-compressed data
    @param Base64-encoded string containing PNG-compressed data
    @return Data stored in an array of size (3, M, N) of type uint8
    """
    fstream = BytesIO(base64.decodebytes(s.encode()))
    im = Image.open(fstream)
    data = np.moveaxis(np.array(im.getdata(), dtype=np.uint8), -1, 0)
    return data.reshape(shape)


def densepose_reencoder(s_in):
    # Decode densepose's compressed output for iuv matrix. Get only bodypart index part. 
    # Then encode back to compressed form. 
    
    fstream = BytesIO(base64.decodebytes(s_in.encode()))
    # option 1:
    #data_iuv = np.asarray(Image.open(fstream))
    # im_out = Image.fromarray(data_iuv[...,0])
    
    # option 2: # only very slightly faster than option 1.  
    im_out = Image.open(fstream).getchannel(0)
    # im_out = Image.open(fstream).split()[0] # option 3. 
    
    fstream = BytesIO()
    im_out.save(fstream, format="png", optimize=True)
    s_out = base64.encodebytes(fstream.getvalue()).decode()
    
    return s_out


def densepose_decoder(s_in):
    fstream = BytesIO(base64.decodebytes(s_in.encode()))
    return np.asarray(Image.open(fstream))


def masks_encoder(arr_):
    # arr_ is a bool matrix of person mask.
    # Then encode back to compressed form. 
    img_dum = arr_.astype(np.int8)
    im_out = Image.fromarray(img_dum)
    
    fstream = BytesIO()
    im_out.save(fstream, format="png", optimize=True)
    s_out = base64.encodebytes(fstream.getvalue()).decode()
    
    return s_out


def get_boundingbox_dense(arr_):
    rows = np.any(arr_, axis=1)
    cols = np.any(arr_, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # [xmin, ymin, xmax, ymax] corresponding to [ left, top, right, bottom ]
    return cmin, rmin, cmax, rmax 


        
def densepose_results2bodypart_layers(result_,img_shape,pred_score_thrs=0.8,keep_nperson=5000):
    bodyparts_layer = np.zeros(img_shape[:2],dtype=np.uint8)
    bodyparts_dum = np.zeros(img_shape[:2],dtype=np.uint8)

    keep_head_box = []
    for instance_id,pred_score in enumerate(result_["dp_scores"]):
        
        if pred_score >= pred_score_thrs and instance_id < keep_nperson:

            pred_box = result_['dp_boxes_xyxy'][instance_id]
            result_encoded = result_['dp_parts_str'][instance_id]
        
            bodyparts_instance = densepose_decoder(result_encoded)
            
            bodyparts_dum[:] = 0
            bodyparts_dum[pred_box[1]:pred_box[1]+bodyparts_instance.shape[0], 
                          pred_box[0]:pred_box[0]+bodyparts_instance.shape[1]] = bodyparts_instance
            
            head_indx_1, head_indx_2 = 23, 24
            head_inds = np.logical_or(bodyparts_dum==head_indx_1,bodyparts_dum==head_indx_2)                 
            
            # against a potential error in finding bounding box. 
            # head_inds[0,:],head_inds[-1,:] = 0, 0
            # head_inds[:,0],head_inds[:,-1] = 0, 0

            # face_borders = [ x_min, y_min, x_max, y_max ] corresponding to [ left, top, right, bottom ]
            if np.sum(head_inds):
                x_min, y_min, x_max, y_max = get_boundingbox_dense(head_inds)
                keep_head_box.append([ x_min, y_min, x_max, y_max, pred_score ])
            
            # used a masking step (bodyparts_dum), otherwise pred_box areas overlap.
            bodyparts_layer[ bodyparts_dum>0 ] = bodyparts_dum[ bodyparts_dum>0 ].copy()

    if np.sum(bodyparts_layer) < 0.1: # ==0: no person detected with pred_score >= pred_score_thrs.
        return None, None
    else:
        return bodyparts_layer,np.asarray(keep_head_box)



def densepose_get_indvheads(result_,img_shape,pred_score_thrs=0.8,keep_nperson=5000):
        
    bodyparts_layer = np.zeros(img_shape[:2],dtype=np.uint8)
    bodyparts_dum = np.zeros(img_shape[:2],dtype=np.uint8)

    keep_head_box = []
    head_num = 1
    for instance_id,pred_score in enumerate(result_["dp_scores"]):
        
        if pred_score >= pred_score_thrs and instance_id < keep_nperson:

            pred_box = result_['dp_boxes_xyxy'][instance_id]
            result_encoded = result_['dp_parts_str'][instance_id]
        
            bodyparts_instance = densepose_decoder(result_encoded)
            
            bodyparts_dum[:] = 0
            bodyparts_dum[pred_box[1]:pred_box[1]+bodyparts_instance.shape[0], 
                          pred_box[0]:pred_box[0]+bodyparts_instance.shape[1]] = bodyparts_instance
            
            head_indx_1, head_indx_2 = 23, 24
            head_inds = np.logical_or(bodyparts_dum==head_indx_1,bodyparts_dum==head_indx_2)                 
            
            # against a potential error in finding bounding box. 
            # head_inds[0,:],head_inds[-1,:] = 0, 0
            # head_inds[:,0],head_inds[:,-1] = 0, 0

            # face_borders = [ x_min, y_min, x_max, y_max ] corresponding to [ left, top, right, bottom ]
            if np.sum(head_inds):
                x_min, y_min, x_max, y_max = get_boundingbox_dense(head_inds)
                keep_head_box.append([ x_min, y_min, x_max, y_max, pred_score ])
            
            # used a masking step (bodyparts_dum), otherwise pred_box areas overlap.
            bodyparts_layer[ head_inds ] = head_num
            head_num += 1

    if np.sum(bodyparts_layer) < 0.1: # ==0: no person detected with pred_score >= pred_score_thrs.
        return None, None
    else:
        return bodyparts_layer,np.asarray(keep_head_box)
    


def densepose_facehand_layers(multiarea_layer):
    
    bodypart_inds = multiarea_layer.copy()
    bodyparts_reduced = np.zeros_like(bodypart_inds)
    
    head_indx_1, head_indx_2 = 23, 24
    head_inds = np.logical_or(bodypart_inds==head_indx_1,bodypart_inds==head_indx_2)
    
    hand_indx_1, hand_indx_2 = 3, 4
    hand_inds = np.logical_or(bodypart_inds==hand_indx_1,bodypart_inds==hand_indx_2)
    
    bodyparts_reduced[head_inds]= 3 # head.
    bodyparts_reduced[hand_inds]= 2 # hands.
    
    bodypart_inds[head_inds | hand_inds] = 0
    bodyparts_reduced[bodypart_inds>0] = 1 # other body-parts. 

    return bodyparts_reduced




    

