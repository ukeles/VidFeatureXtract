#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Semantic segmentation on video frames

"""

import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

from ..utils.io_helpers import check_outputfile, array2str_encoder, finalize_and_save
from .vis_helper import Visualizer # looks better...
# from .vis_simple import visualize_segments # slightly faster than vis_helper.Visualizer()

def extract_oneformer_segmentation(vr, output_dir, model_name='oneformer_coco_swin_large',
                                   overwrite_ok=False, saveviz=True):
    """
    Performs semantic segmentation on frames from a video using the OneFormer model, 
    saves extracted data, and optionally visualizes results in an output video.

    see:
        https://arxiv.org/abs/2211.06220
        https://github.com/SHI-Labs/OneFormer

    Parameters
    ----------
    vr : VideoReader
        A VideoReader object providing access to video frames.
    output_dir : str
        The directory where output files will be saved.
    model_name: str
        The pre-trained model for densepose.
    overwrite_ok : bool, optional
        If True, existing files will be overwritten. Default is False.
    saveviz : bool, optional
        If True, a visualization of the detection will be saved as a video. Default is True.
        
    Returns
    -------
    None
    """

    if 'coco' in model_name:
        from .label_cfgs import COCO133_CATEGORIES
        categs_all = np.asarray(COCO133_CATEGORIES)
        file_ext = 'coco'
    elif 'ade20k' in model_name:
        from .label_cfgs import ADE150_CATEGORIES
        categs_all = np.asarray(ADE150_CATEGORIES)
        file_ext = 'ade'


    nframes = vr.frame_count  # Number of frames in the video.
    extract_frames = vr.extraction_frames

    if extract_frames is not None:
        print(f'Performing segmentation for {len(extract_frames)} of {nframes} total frames...\n')
        vid_basename = f'{vr.basename}_oneformer{file_ext}_fps_{vr.extraction_fps}'
    else:
        vid_basename = f'{vr.basename}_oneformer{file_ext}'

    frame_width, frame_height = vr.resolution  # Resolution of the video (width, height).
    vid_fps = vr.fps # Frames per second of the video.
    
    # Output file paths
    outfile_pkl = os.path.join(output_dir, f'{vid_basename}.pkl')
    
    # Check if output files already exist
    check_outputfile(outfile_pkl, overwrite_ok)
    
    # Available pre-trained models: oneformer_coco_swin_large , oneformer_ade20k_swin_large
    processor = OneFormerProcessor.from_pretrained(f"shi-labs/{model_name}")
    model = OneFormerForUniversalSegmentation.from_pretrained(f"shi-labs/{model_name}")
    
    output_vid_file = None
    if saveviz:
        if extract_frames is not None:
            output_dir_frames = os.path.join(output_dir,f'{vid_basename}')
            os.makedirs(output_dir_frames, exist_ok=True)
            print('\nSaving visualization for the extracted frames only...\n')
        else:
            output_fname = os.path.join(output_dir, f'{vid_basename}.mp4')
            output_vid_file = cv2.VideoWriter(filename=output_fname,
                                              fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                                              fps=float(vid_fps),
                                              frameSize=(frame_width, frame_height),
                                              isColor=True)

    feats_data = {}
    for fii, frame_ii in enumerate(tqdm(vr, total=nframes)):
        
        # slow but in general a safer way for loading specific frames
        if extract_frames is not None and fii not in extract_frames:
            continue
        
        # RGB works better then BGR -- use frame_ii directly
        inputs = processor(frame_ii, task_inputs=["semantic"], return_tensors="pt")
    
        # bactching does not improve speed. 
        with torch.no_grad():
            outputs = model(**inputs)
    
        predicted_semantic_map = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(frame_height,frame_width)] )[0]
        
        sem_seg = predicted_semantic_map.detach().cpu().numpy()
    
        if np.any(sem_seg):
            feats_data[f'frame_{fii+1}'] = array2str_encoder(sem_seg)
            
        # Visualization part
        if saveviz:
            visualizer = Visualizer(frame_ii)
            viz_img = visualizer.draw_sem_seg(sem_seg, categs_all, alpha=0.6)
            # img = visualize_segments(frame_ii, sem_seg, categs_all, alpha=0.6)
            if extract_frames is not None:
                viz_img.save(os.path.join(output_dir_frames, f'frame_{fii}.png'))
                # cv2.imwrite(os.path.join(output_dir_frames, f'frame_{fii}.png'), img)
            else:                
                output_vid_file.write(viz_img.get_image()[...,::-1])
                # output_vid_file.write(img)
    
    # Finalize video writing and save feature data
    finalize_and_save(output_vid_file, outfile_pkl, feats_data)

