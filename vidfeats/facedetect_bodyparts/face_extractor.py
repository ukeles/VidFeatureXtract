#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Face detection and exctraction of some face-related features from videos

"""

import os
import cv2
import numpy as np
from tqdm import tqdm

from ..utils.io_helpers import check_outputfile
from .facebody_helpers import faces2dets, get_faceareas_simple, get_faceareas
from .facebody_helpers import visualize_facedetections, finalize_and_save

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
    app = FaceAnalysis(allowed_modules=['detection'],
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

