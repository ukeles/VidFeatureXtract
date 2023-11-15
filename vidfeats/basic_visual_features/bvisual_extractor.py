#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions for extracting basic visual features

"""

import os
import cv2
import numpy as np
from tqdm import tqdm

from ..utils.io_helpers import check_outputfile
from ..utils.video_processing import compute_downsample_size


# ----- RGB-HSV-Luminance features -----
def extract_colors(vr, output_dir, overwrite_ok=False):
    """
    Extract RGB, HSV, and Luminance features from video frames, 
    as well as their frame-to-frame differences (delta).

    Parameters:
    -----------
    vr : object
        Video reader object.
    output_dir : str
        Directory to save the extracted features.
    overwrite_ok : bool, optional
        If True, allows overwriting existing files. Default is False.

    Returns:
    --------
    None
        Saves two .npy files in the output directory. One for the feature values 
        and another for the deltas.
    """
    
    print('---> Note that the resizing of video or batching are not implemented for color features\n')
    
    nframes = vr.frame_count # Number of frames in the video.
    resolution_wh = vr.resolution # Resolution of the video (width, height).
    basename = vr.basename # Base name of the video file.
    
    # Calculate the total number of pixels in a frame
    num_pixels = np.prod(resolution_wh, dtype=float)
    
    # Construct output filenames
    outfile_vals = os.path.join(output_dir, f'{basename}_rgbhsvl_vals.npy')
    outfile_delta = os.path.join(output_dir, f'{basename}_rgbhsvl_delta.npy')
    
    # Check if output files already exist
    check_outputfile(outfile_vals, overwrite_ok)
    check_outputfile(outfile_delta, overwrite_ok)
    
    # Initialize arrays to store feature values and deltas
    keep_vals = np.zeros((nframes, 7))
    keep_delta = np.zeros((nframes, 7))
    
    # Iterate over each frame to extract features
    for fii, frame_rgb in enumerate(tqdm(vr, total=nframes)):
        
        # For the first frame, initialize previous frame values to zero
        if fii == 0:
            frame_pre_rgb = np.zeros(frame_rgb.shape)
            frame_pre_hsv = np.zeros(frame_rgb.shape)
            frame_pre_lum = np.zeros((frame_rgb.shape[0], frame_rgb.shape[1]))
    
        # Convert the frame from RGB to HSV and LAB (for luminance)
        frame_hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV).astype(float)  # We start from RGB.
        frame_lum = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)[:,:,0].astype(float)
        
        # Calculate the frame-to-frame difference (delta) for each channel
        delta_rgb = np.abs(frame_rgb - frame_pre_rgb).sum(0).sum(0) / num_pixels
        delta_hsv = np.abs(frame_hsv - frame_pre_hsv).sum(0).sum(0) / num_pixels
        delta_lum = np.abs(frame_lum - frame_pre_lum).sum() / num_pixels
        keep_delta[fii,:] = np.r_[delta_rgb, delta_hsv, delta_lum]
        
        # Calculate the mean value for each channel in the current frame
        rgb_vals = frame_rgb.sum(0).sum(0) / num_pixels
        hsv_vals = frame_hsv.sum(0).sum(0) / num_pixels
        lum_vals = frame_lum.sum() / num_pixels
        keep_vals[fii,:] = np.r_[rgb_vals, hsv_vals, lum_vals]
        
        # Update the 'previous frame' variables for the next iteration
        frame_pre_rgb = frame_rgb.copy()
        frame_pre_hsv = frame_hsv.copy()
        frame_pre_lum = frame_lum.copy()
    
    # Assert the frame resolution is as expected
    assert (frame_rgb.shape[1], frame_rgb.shape[0]) == resolution_wh
    
    # Save the extracted feature values and deltas to .npy files
    np.save(outfile_vals, keep_vals)
    np.save(outfile_delta, keep_delta)




# ----- GIST (Global Image Statistics) Feature Extraction -----
# References:
# 1. https://people.csail.mit.edu/torralba/code/spatialenvelope/
# 2. https://doi.org/10.1016/S0079-6123(06)55002-2
from .pygist import LMgist
import multiprocessing as mp

def extract_gist(vr, params, output_dir, overwrite_ok=False, outfile_txt=None):
    """
    Extract GIST features from video frames.

    Parameters:
    -----------
    vr : object
        Video reader object.
    params : dict
        Dictionary of GIST extraction parameters.
    output_dir : str
        Directory to save the extracted GIST features.
    overwrite_ok : bool, optional
        If True, allows overwriting existing files. Default is False.
    outfile_txt : str, optional
        Custom text to append to output filename. Default is None.

    Returns:
    --------
    None
        Saves GIST features to a .npy file.
    """

    nframes = vr.frame_count # Number of frames in the video.
    resolution_wh = vr.resolution # Resolution of the video (width, height).
    basename = vr.basename # Base name of the video file.

    # Compute the downsampling size based on input parameters
    down_use = compute_downsample_size(params['image_size'], resolution_wh)

    # Set the output filename based on the provided parameters
    if outfile_txt is None:
        output_file = os.path.join(output_dir, f'{basename}_gist{down_use[0]}x{down_use[1]}.npy')
    else:
        # In case we extract and save feature with parameters other than default ones.
        output_file = os.path.join(output_dir, f'{basename}_{outfile_txt}.npy')

    # Check if output file exists and handle accordingly
    check_outputfile(output_file, overwrite_ok)

    # Use default GIST parameters (as the original MATLAB implementation) if not provided by user
    params.setdefault('orientationsPerScale', [8, 8, 8, 8])
    params.setdefault('numberBlocks', [4, 4])
    params.setdefault('fc_prefilt', 4)
    params.setdefault('boundaryExtension', 32)

    # Initialize a Gist object with the given parameters
    get_gist = LMgist(params, img_size_wh=down_use)

    # Define number of batches for feature extraction
    nbatches = params['nbatches']  # Bypass batching with using nbatches = 1.
    batch_size = int(np.ceil(nframes/nbatches))
    
    # Extract GIST features in batches
    gist_features = []
    for bdx in range(nbatches):
        start_frame, end_frame = batch_size * bdx, min(batch_size * (bdx + 1), nframes)

        if nbatches > 1:
            print(f'\nBatch {bdx + 1}/{nbatches} [{start_frame}, {end_frame - 1}]')

        print('Converting to gray scale...')
        # the original matlab code first converts to gray scale than resizes:
        # Convert frames to grayscale and resize them
        stimulus_batch = [
            cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),  # Convert to grayscale
                down_use, 
                interpolation=cv2.INTER_LINEAR
            )
            for frame in tqdm(vr[start_frame:end_frame])
        ]        
        
        # Extract GIST features using multiprocessing for improved speed
        print('Extracting features...')
        with mp.Pool(mp.cpu_count()) as pool:
            p = pool.imap(get_gist.gist_extract, stimulus_batch)  # Map the extraction function to frames
            batched_responses = list(tqdm(p, total=len(stimulus_batch)))
        
        gist_features.append(batched_responses)
    
    # Save extracted GIST features to an output .npy file
    np.save(output_file, np.vstack(gist_features))




# ----- Motion-energy [spatio-temporal Gabor filters] feature extraction -----
# These features capture the motion information in a video by applying 
# Gabor filters across space and time.
# For more details:
# 1. GitHub repository: https://github.com/gallantlab/pymoten
# 2. Official documentation: https://gallantlab.org/pymoten/
def extract_moten(vr, params, output_dir, overwrite_ok=False, outfile_txt=None):
    """
    Extract motion energy features using spatio-temporal Gabor filters.

    Parameters:
    -----------
    vr : object
        Video reader object.
    params : dict
        Dictionary of moten extraction parameters.
    output_dir : str
        Directory to save the extracted moten features.
    overwrite_ok : bool, optional
        If True, allows overwriting existing files. Default is False.
    outfile_txt : str, optional
        Custom text to append to output filename. Default is None.

    Returns:
    --------
    None
        Saves moten features to a .npy file.

    Note:
    - If 'add_noise' is present in params and set to True, 
      the results may slightly differ on each run.
    """

    # An updated copy of pymoten library (https://github.com/gallantlab/pymoten)
    # is located under vidfeats.thirdparty folder.
    from ..thirdparty.moten.pyramids import MotionEnergyPyramid
    from ..thirdparty.moten.io import imagearray2luminance
    
    nframes = vr.frame_count # Number of frames in the video.
    resolution_wh = vr.resolution # Resolution of the video (width, height).
    basename = vr.basename # Base name of the video file.
    vid_fps = vr.fps # Frames per second of the video.
    
    # Set default preprocessing pipeline to 'moten' if not specified
    params.setdefault('preprocess_pipeline', 'moten')
    assert params['preprocess_pipeline'] in ['moten', 'opencv']

    add_txt = '_cv' if params['preprocess_pipeline'] == 'opencv' else ''
    
    # Set default for adding noise to True if not specified
    params.setdefault('add_noise', True)
    
    if params['add_noise']:
        add_txt += '_ns' 
        noise = 0.1 

    down_use = compute_downsample_size(params['image_size'], resolution_wh)

    # Determine the name for the output file
    outfile_suffix = outfile_txt if (outfile_txt is not None) else f"moten{down_use[0]}x{down_use[1]}"
    output_file = os.path.join(output_dir, f'{basename}_{outfile_suffix}{add_txt}.npy')
    
    check_outputfile(output_file, overwrite_ok)

    # Define the Gabor pyramid for motion energy extraction
    pyramid = MotionEnergyPyramid(
        stimulus_vhsize=down_use[::-1],  # requires (vdim, hdim)
        stimulus_fps=round(vid_fps),  # requires an integer
        # spatial_frequencies=[0, 2, 4, 8, 16, 32],
    )
    
    window = int(np.ceil(pyramid.definition['filter_temporal_width'] / 2))
    
    # Define number of batches for feature extraction
    nbatches = params['nbatches']  # Bypass batching with using nbatches = 1.
    batch_size = int(np.ceil(nframes/nbatches))
    
    # Extract moten features in batches
    moten_features = []
    for bdx in range(nbatches):
        start_frame, end_frame = batch_size * bdx, min(batch_size * (bdx + 1), nframes)
        
        if nbatches > 1:
            print(f'\nBatch {bdx+1}/{nbatches} [{start_frame}, {end_frame-1}]')
    
        # Calculate padding for batches
        batch_start = max(start_frame - window, 0)
        batch_end = min(end_frame + window, nframes)

        print('Converting to luminance...')
        if params['preprocess_pipeline'] == 'moten':
            stimulus_batch = imagearray2luminance(
                np.array(vr[batch_start:batch_end]),
                size=down_use[::-1]
            )
        elif params['preprocess_pipeline'] == 'opencv':
            stimulus_batch = [
                cv2.cvtColor(
                    cv2.resize(frame, down_use, interpolation=cv2.INTER_LINEAR),
                    cv2.COLOR_RGB2LAB
                )[..., 0] * 100.0 / 255.0
                for frame in tqdm(vr[batch_start:batch_end])
            ]
            stimulus_batch = np.asarray(stimulus_batch)
        
        
        # elif params['preprocess_pipeline'] == 'opencv':
        #     stimulus_batch = []
        #     for frame_ii_rgb in tqdm(vr[batch_start:batch_end]):
                
        #         frame_ii_lum = cv2.resize(frame_ii_rgb, down_use,
        #                                   interpolation=cv2.INTER_LINEAR,
        #                                   # interpolation=cv2.INTER_LANCZOS4, # looks similar to linear indeed.
        #                                   )
        
        #         # --- RGB2LAB is a slow conversion. Resizing image first helps to speed up ---
        #         # from skimage.color import rgb2lab # is slightly faster than moten implementation
        #         # use the moten original.
        #         # frame_ii_lum = moten.colorspace.rgb2lab(frame_ii_lum/255.)[..., 0] # the same as skimage.color.rgb2lab()
        #         # frame_ii_lum = rgb2lab(frame_ii_lum)[..., 0] # 
        #         frame_ii_lum = cv2.cvtColor(frame_ii_lum, cv2.COLOR_RGB2LAB)[..., 0]*100.0/255.0 # close appx., but faster.
        #         stimulus_batch.append(frame_ii_lum)
                
        #     stimulus_batch = np.asarray(stimulus_batch)        
        
        
        # Add noise if specified to deal with constant black areas;
        # https://gallantlab.github.io/voxelwise_tutorials/_auto_examples/shortclips/07_extract_motion_energy.html#compute-the-motion-energy
        if params['add_noise']:
            stimulus_batch += np.random.randn(*stimulus_batch.shape) * noise
            stimulus_batch = np.clip(stimulus_batch, 0, 100)
        
        batched_responses = pyramid.project_stimulus(stimulus_batch)
        
        # Trim edges of the batched responses based on the current batch index
        if nbatches > 1:
            if bdx == 0:
                batched_responses = batched_responses[:-window]
            elif bdx + 1 == nbatches:
                batched_responses = batched_responses[window:]
            else:
                batched_responses = batched_responses[window:-window]
        
        moten_features.append(batched_responses)
    
    # Save extracted moten features to an output .npy file
    np.save(output_file, np.vstack(moten_features))

