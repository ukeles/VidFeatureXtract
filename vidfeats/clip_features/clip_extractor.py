#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


"""


import os
import open_clip
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

import cv2

from ..utils.io_helpers import check_outputfile

def extract_clip_features(
        vr: object,
        params: dict, 
        output_dir: str,
        overwrite_ok: bool = False,
        ):
    """
    Extract CLIP image features from video frames.

    Parameters:
    -----------
    vr : object
        Video reader object.
    params : dict
        Dictionary of open_clip extraction parameters.
    output_dir : str
        Directory to save the extracted Clip features.
    overwrite_ok : bool, optional
        If True, allows overwriting existing files. Default is False.

    Returns:
    --------
    None
        Saves a .npy file in the output directory.
    """

    device = params.get('device')
    arch = params.get('arch')
    pretrained = params.get('pretrained')
    image_resize_mode = params.get('image_resize_mode')

    # Resolve device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Load model and preprocessing pipeline
    model, _, preprocess = open_clip.create_model_and_transforms(
        arch, pretrained=pretrained, 
        image_resize_mode=image_resize_mode, 
        device=device
        )
    model.eval()

    nframes = vr.frame_count # Number of frames in the video.
    extract_frames = vr.extraction_frames

    if extract_frames is not None:
        print(f'Performing feature extraction for {len(extract_frames)} of {nframes} total frames...\n')
        vid_basename = f'{vr.basename}_fps_{vr.extraction_fps}'
    else:
        vid_basename = vr.basename
        
    outfile_feats = os.path.join(output_dir, 
         f'{vid_basename}_{arch}_{pretrained}_{image_resize_mode[:2]}_features.npy')

    # Check if output files already exist
    check_outputfile(outfile_feats, overwrite_ok)
    
    
    if params.get('show_progress'):
        iter_func = tqdm
    else:
        def no_op_iter(x, total=None):  # No-operation iterator
            return x
        iter_func = no_op_iter
    
    
    features = []
    with torch.no_grad():
        for fii, frame_ii in enumerate(iter_func(vr, total=nframes)):

            # slow but in general a safer way for loading specific frames
            if extract_frames is not None and fii not in extract_frames:
                continue
                
            # Load and preprocess image
            image = Image.fromarray(frame_ii).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)

            #-------------
            # test_img = image_tensor.squeeze(0).cpu().numpy()
            # import matplotlib.pylab as plt
            # plt.imshow(test_img[0,...])
            #-------------

            # Encode and convert to numpy
            feat = model.encode_image(image_tensor)
            
            # Normalize if not already normalized  --- keep as it is for now
            # feat /= feat.norm(dim=-1, keepdim=True)

            features.append(feat.cpu().numpy().squeeze(0))

    # Stack into array
    features_array = np.vstack(features)

    # Save extracted features to an output .npy file
    np.save(outfile_feats, features_array)



import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as pe_transforms


def extract_pe_features(
        vr: object,
        params: dict, 
        output_dir: str,
        overwrite_ok: bool = False,
        ):
    """
    Extract CLIP image features from video frames.

    Parameters:
    -----------
    vr : object
        Video reader object.
    params : dict
        Dictionary of Facebook Perception Models extraction parameters.
    output_dir : str
        Directory to save the extracted Clip features.
    overwrite_ok : bool, optional
        If True, allows overwriting existing files. Default is False.

    Returns:
    --------
    None
        Saves a .npy file in the output directory.
    """

    device = params.get('device')
    model_name = params.get('model_name')
    image_resize_mode = params.get('image_resize_mode')

    if image_resize_mode == 'squash': # default in `pe_transforms`
        center_crop = False
        full_frame = False
    elif image_resize_mode == 'shortest':
        center_crop = True
        full_frame = False
    elif image_resize_mode == 'longest':
        center_crop = True
        full_frame = True
        
    # Resolve device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Load PE-Core model and preprocessing
    model = pe.CLIP.from_config(model_name, pretrained=True).to(device) # Downloads from HF
    model.eval()
    preprocess = pe_transforms.get_image_transform(model.image_size, center_crop)
    
    nframes = vr.frame_count # Number of frames in the video.
    extract_frames = vr.extraction_frames

    if extract_frames is not None:
        print(f'Performing feature extraction for {len(extract_frames)} of {nframes} total frames...\n')
        vid_basename = f'{vr.basename}_fps_{vr.extraction_fps}'
    else:
        vid_basename = vr.basename
        
    outfile_feats = os.path.join(output_dir, 
                 f'{vid_basename}_{model_name}_{image_resize_mode[:2]}_features.npy')

    # Check if output files already exist
    check_outputfile(outfile_feats, overwrite_ok)
    
    if params.get('show_progress'):
        iter_func = tqdm
    else:
        def no_op_iter(x, total=None):  # No-operation iterator
            return x
        iter_func = no_op_iter
    
    
    features = []
    with torch.no_grad():
        for fii, frame_ii in enumerate(iter_func(vr, total=nframes)):

            # slow but in general a safer way for loading specific frames
            if extract_frames is not None and fii not in extract_frames:
                continue
            
            # keep these calculations inside the vr loop,  
            # just in case the resolution is variable
            if full_frame:
                h, w = frame_ii.shape[:2]
                size = max(h, w)
                top = (size - h) // 2
                bottom = size - h - top
                left = (size - w) // 2
                right = size - w - left
            
                frame_ii = cv2.copyMakeBorder(
                    frame_ii, top, bottom, left, right, cv2.BORDER_CONSTANT, 
                    value=(0, 0, 0) # black
                )
                

            # Load and preprocess image
            image = Image.fromarray(frame_ii).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)

            #-------------
            # test_img = image_tensor.squeeze(0).cpu().numpy()
            # import matplotlib.pylab as plt
            # plt.imshow(test_img[0,...])
            #-------------

            # Encode and convert to numpy
            feat = model.encode_image(image_tensor)
            
            # Normalize if not already normalized  --- keep as it is for now
            # feat /= feat.norm(dim=-1, keepdim=True)

            features.append(feat.cpu().numpy().squeeze(0))

    # Stack into array
    features_array = np.vstack(features)

    # Save extracted features to an output .npy file
    np.save(outfile_feats, features_array)


