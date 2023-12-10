#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Face detection and exctraction of some face-related features from videos

"""

import os
import cv2
import numpy as np
from tqdm import tqdm

from ..utils.io_helpers import check_outputfile, finalize_and_save
from .facebody_helpers import faces2dets, get_faceareas_simple, get_faceareas
from .facebody_helpers import YOLOv8_face, visualize_facedetections


def extract_faces_insight(vr, output_dir, overwrite_ok=False, saveviz=True, det_thresh=0.5):
    """
    Extracts faces from a video using the insightface tool, saves extracted data, 
    and optionally visualizes results in an output video.

    Parameters
    ----------
    vr : VideoReader
        A VideoReader object providing access to video frames.
    output_dir : str
        The directory where output files will be saved.
    overwrite_ok : bool, optional
        If True, existing files will be overwritten. Default is False.
    saveviz : bool, optional
        If True, a visualization of the detection will be saved as a video. Default is True.
    det_thresh : float, optional
        The threshold for face detection confidence. Faces with detection scores below this 
        threshold will not be considered. Default is 0.5.   
        
    Returns
    -------
    None
    """

    # Check for the presence of required modules.
    try:
        from insightface.app import FaceAnalysis
    except:
        txt = "Required module 'insightface' not found.\n"\
        +"Please install from 'https://github.com/deepinsight/insightface'"
        raise ModuleNotFoundError(txt)
    
    
    import onnxruntime as ort
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        print('\n\n\nNote that ONNX Runtime was not installed with GPU support.')
        print('You might consider reinstalling it with:')
        print('pip install onnxruntime-gpu -U')
        print('\n\n')
    
    nframes = vr.frame_count  # Number of frames in the video.
    frame_width, frame_height = vr.resolution  # Resolution of the video (width, height).
    vid_fps = vr.fps # Frames per second of the video.
    vid_basename = f'{vr.basename}_insightface_thresh{det_thresh}'
    
    # Output file paths
    outfile_pkl = os.path.join(output_dir, f'{vid_basename}.pkl')
    outfile_arr = os.path.join(output_dir, f'{vid_basename}.npy')
    
    # Check if output files already exist
    check_outputfile(outfile_pkl, overwrite_ok)
    check_outputfile(outfile_arr, overwrite_ok)
    
    # Initialize FaceAnalysis with detection model only
    app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'],
                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=det_thresh) # default det_thresh=0.5
    
    # Calculate the total number of pixels in a frame
    num_pixels = frame_width * frame_height
    
    output_vid_file = None
    if saveviz:
        output_fname = os.path.join(output_dir, f'{vid_basename}.mp4')
        output_vid_file = cv2.VideoWriter(filename=output_fname,
                                          fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                                          fps=float(vid_fps),
                                          frameSize=(frame_width, frame_height),
                                          isColor=True)
    
    feats_data = {}
    feats_arr = np.zeros((nframes, 3))
    for fii, frame_ii in enumerate(tqdm(vr, total=nframes)):
        # Convert image format from RGB to BGR for insightface
        img = cv2.cvtColor(frame_ii, cv2.COLOR_RGB2BGR)
        faces = app.get(img)
        dets = faces2dets(faces)
        
        if len(dets) > 0: # at least one face detected
            feats_data[f'frame_{fii+1}'] = dets
            face_areas_nums, _ = get_faceareas_simple(dets, frame_height, frame_width,
                                                      detection_thrs=det_thresh)
            face_areas, eyes_areas, mouth_areas, face_boxes, face_boxes_eyes, face_boxes_mouth = get_faceareas(
                dets, frame_height, frame_width, detection_thrs=det_thresh, return_mask=True, return_box=True)
            
            diff_faces = np.unique(face_areas_nums[face_areas_nums > 0])
            avg_of_face_areas = np.mean([(face_areas_nums == face_ii).sum() / num_pixels for face_ii in diff_faces])
            feats_arr[fii, :] = [len(dets), np.sum(face_areas) / num_pixels, avg_of_face_areas]
        
        # Visualization part
        if saveviz:
            if len(dets) > 0: # at least one face detected
                visualize_facedetections(img, face_boxes, face_boxes_eyes, face_boxes_mouth)
            
            output_vid_file.write(img)
    
    # Finalize video writing and save feature data
    finalize_and_save(output_vid_file, outfile_pkl, feats_data, outfile_arr, feats_arr)


def extract_faces_yolov8face(vr, output_dir, overwrite_ok=False, saveviz=True, det_thresh=0.3):
    """
    Extracts faces from a video using the Yolov8-face tool, saves extracted data, 
    and optionally visualizes results in an output video.
    
    adapted from:
        https://github.com/derronqi/yolov8-face

    Parameters
    ----------
    vr : VideoReader
        A VideoReader object providing access to video frames.
    output_dir : str
        The directory where output files will be saved.
    overwrite_ok : bool, optional
        If True, existing files will be overwritten. Default is False.
    saveviz : bool, optional
        If True, a visualization of the detection will be saved as a video. Default is True.
    det_thresh : float, optional
        The threshold for face detection confidence. Faces with detection scores below this 
        threshold will not be considered. Default is 0.3.   
        
    Returns
    -------
    None
    """
    from ultralytics import YOLO
    

    nframes = vr.frame_count  # Number of frames in the video.
    frame_width, frame_height = vr.resolution  # Resolution of the video (width, height).
    vid_fps = vr.fps # Frames per second of the video.
    vid_basename = f'{vr.basename}_yolov8face_thresh{det_thresh}'
    
    # Output file paths
    outfile_pkl = os.path.join(output_dir, f'{vid_basename}.pkl')
    outfile_arr = os.path.join(output_dir, f'{vid_basename}.npy')
    
    # Check if output files already exist
    check_outputfile(outfile_pkl, overwrite_ok)
    check_outputfile(outfile_arr, overwrite_ok)
    
    # Initialize face detection model
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    modelpath = os.path.join(curr_dir,'yolov8_pretrained','yolov8n-face.pt')
    face_detector = YOLO(modelpath)

    # Calculate the total number of pixels in a frame
    num_pixels = frame_width * frame_height
    
    output_vid_file = None
    if saveviz:
        output_fname = os.path.join(output_dir, f'{vid_basename}.mp4')
        output_vid_file = cv2.VideoWriter(filename=output_fname,
                                          fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                                          fps=float(vid_fps),
                                          frameSize=(frame_width, frame_height),
                                          isColor=True)
    
    feats_data = {}
    feats_arr = np.zeros((nframes, 3))
    for fii, frame_ii in enumerate(tqdm(vr, total=nframes)):
        # Convert image format from RGB to BGR for yolov8-face model | works best this ways.
        img = cv2.cvtColor(frame_ii, cv2.COLOR_RGB2BGR)
        
        results = face_detector.predict(img, conf=det_thresh, imgsz=640,  
                                        iou=0.5, max_det=5000, verbose=False)

        result = results[0]
        box_data = result.boxes.data.cpu().numpy()
        keyp_xy_3d = result.keypoints.xy.cpu().numpy()
        keyp_xy = keyp_xy_3d.reshape(*keyp_xy_3d.shape[:-2], -1)
        dets = np.hstack((box_data[:,:-1], keyp_xy))

        if len(dets) > 0: # at least one face detected
            feats_data[f'frame_{fii+1}'] = dets
            face_areas_nums, _ = get_faceareas_simple(dets, frame_height, frame_width,
                                                      detection_thrs=det_thresh)
            face_areas, eyes_areas, mouth_areas, face_boxes, face_boxes_eyes, face_boxes_mouth = get_faceareas(
                dets, frame_height, frame_width, detection_thrs=det_thresh, return_mask=True, return_box=True)
            
            diff_faces = np.unique(face_areas_nums[face_areas_nums > 0])
            avg_of_face_areas = np.mean([(face_areas_nums == face_ii).sum() / num_pixels for face_ii in diff_faces])
            feats_arr[fii, :] = [len(dets), np.sum(face_areas) / num_pixels, avg_of_face_areas]
        
        # Visualization part
        if saveviz:
            if len(dets) > 0: # at least one face detected
                visualize_facedetections(img, face_boxes, face_boxes_eyes, face_boxes_mouth)
            
            output_vid_file.write(img)
    
    # Finalize video writing and save feature data
    finalize_and_save(output_vid_file, outfile_pkl, feats_data, outfile_arr, feats_arr)


def extract_faces_yolov8face_cv(vr, output_dir, overwrite_ok=False, saveviz=True, det_thresh=0.3):
    """
    Extracts faces from a video using the Yolov8-face and opencv, saves extracted data, 
    and optionally visualizes results in an output video.
    
    adapted from:
        https://github.com/derronqi/yolov8-face
        https://github.com/hpc203/yolov8-face-landmarks-opencv-dnn

    Parameters
    ----------
    vr : VideoReader
        A VideoReader object providing access to video frames.
    output_dir : str
        The directory where output files will be saved.
    overwrite_ok : bool, optional
        If True, existing files will be overwritten. Default is False.
    saveviz : bool, optional
        If True, a visualization of the detection will be saved as a video. Default is True.
    det_thresh : float, optional
        The threshold for face detection confidence. Faces with detection scores below this 
        threshold will not be considered. Default is 0.3.   
        
    Returns
    -------
    None
    """

    nframes = vr.frame_count  # Number of frames in the video.
    frame_width, frame_height = vr.resolution  # Resolution of the video (width, height).
    vid_fps = vr.fps # Frames per second of the video.
    vid_basename = f'{vr.basename}_yolov8face_cv_thresh{det_thresh}'
    
    # Output file paths
    outfile_pkl = os.path.join(output_dir, f'{vid_basename}.pkl')
    outfile_arr = os.path.join(output_dir, f'{vid_basename}.npy')
    
    # Check if output files already exist
    check_outputfile(outfile_pkl, overwrite_ok)
    check_outputfile(outfile_arr, overwrite_ok)
    
    # Initialize face detection model
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    modelpath = os.path.join(curr_dir,'yolov8_pretrained','yolov8n-face.onnx')
    face_detector = YOLOv8_face(modelpath, conf_thres=det_thresh, iou_thres=0.5)

    # Calculate the total number of pixels in a frame
    num_pixels = frame_width * frame_height
    
    output_vid_file = None
    if saveviz:
        output_fname = os.path.join(output_dir, f'{vid_basename}.mp4')
        output_vid_file = cv2.VideoWriter(filename=output_fname,
                                          fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                                          fps=float(vid_fps),
                                          frameSize=(frame_width, frame_height),
                                          isColor=True)
    
    feats_data = {}
    feats_arr = np.zeros((nframes, 3))
    for fii, frame_ii in enumerate(tqdm(vr, total=nframes)):
        # Convert image format from RGB to BGR for yolov8-face model
        img = cv2.cvtColor(frame_ii, cv2.COLOR_RGB2BGR)
        
        boxes, scores, classids, kpts = face_detector.detect(img)
        dets = np.hstack((boxes,scores,kpts))
        
        if len(dets) > 0: # at least one face detected
            feats_data[f'frame_{fii+1}'] = dets
            face_areas_nums, _ = get_faceareas_simple(dets, frame_height, frame_width,
                                                      detection_thrs=det_thresh)
            face_areas, eyes_areas, mouth_areas, face_boxes, face_boxes_eyes, face_boxes_mouth = get_faceareas(
                dets, frame_height, frame_width, detection_thrs=det_thresh, return_mask=True, return_box=True)
            
            diff_faces = np.unique(face_areas_nums[face_areas_nums > 0])
            avg_of_face_areas = np.mean([(face_areas_nums == face_ii).sum() / num_pixels for face_ii in diff_faces])
            feats_arr[fii, :] = [len(dets), np.sum(face_areas) / num_pixels, avg_of_face_areas]
        
        # Visualization part
        if saveviz:
            if len(dets) > 0: # at least one face detected
                visualize_facedetections(img, face_boxes, face_boxes_eyes, face_boxes_mouth)
            
            output_vid_file.write(img)
    
    # Finalize video writing and save feature data
    finalize_and_save(output_vid_file, outfile_pkl, feats_data, outfile_arr, feats_arr)


