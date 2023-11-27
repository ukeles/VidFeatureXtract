#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""


import os
import argparse
import numpy as np

from vidfeats.mediaio import Video
from vidfeats.utils.io_helpers import str2bool

from vidfeats.basic_visual_features.bvisual_extractor import extract_colors, extract_gist, extract_moten
from vidfeats.facedetect_bodyparts.face_extractor import extract_faces_insight


# List of available features for extraction
features_base = ['getinfo', 'count_frames']
features_extract = ['colors', 'gist', 'moten', 'face_insg']
features_list = features_base + features_extract


def run_feature_extraction(inputs):
    """
    Extracts specified visual features from the given video.

    Parameters:
    -----------
    inputs : dict
        A dictionary containing the required inputs for feature extraction, including:
        - 'video' : str, path to the video file.
        - 'feature' : str, name of the feature to be extracted.
        - 'output_dir' : str, directory where the extracted features should be saved.
        - 'overwrite' : bool, flag indicating if existing files should be overwritten.
        - 'resfac' : int, resizing factor for the video.
        - 'width' : int, target width for video resizing.
        - 'height' : int, target height for video resizing.
        - 'nbatches' : int, number of batches (used in potential batching scenarios).
        - 'saveviz' : bool, flag indicating whether to save the visualization of detections

    Returns:
    --------
    None
    """
    # Extracting input parameters
    video_file = inputs['video']
    feature_name = inputs['feature']
    output_dir = inputs['output_dir']
    overwrite_ok = inputs['overwrite']
    resize_factor = inputs['resfac']
    image_size = (inputs.get('width',None), inputs.get('height',None))
    nbatches = inputs['nbatches']
    saveviz = inputs['saveviz']

    # Create a Video object
    vr = Video(video_file)

    # Display video metadata
    print(f'Video resolution: {vr.resolution}')
    print(f'Video # of frames and fps: {vr.frame_count}, {vr.fps}')

    # Determine if image resizing is needed based on provided inputs
    if image_size[0] is None and image_size[1] is None:
        if resize_factor==1:
            image_size = vr.resolution
            print(f'Using the original video size of {image_size}\n')
        else:
            image_size = (vr.resolution[0]//resize_factor, vr.resolution[1]//resize_factor)
            print(f'Resizing the video to {image_size} using the resizing factor={resize_factor}\n')
    else:
        if None in image_size:
            raise SystemExit('To resize the video using width and height, '+
                             'please provide both these parameters')
        print(f"Resizing the video to {image_size} using width ({inputs['width']}) " +
              f"and height ({inputs['height']}) provided\n")

    # Return if only video info is requested
    if feature_name == 'getinfo':
        return

    # Return after examining number of frames in the video
    if feature_name == 'count_frames':
        vr.examine_nframes()
        return

    # ----- Extract RGB-HSV-Luminance features from the video -----
    if feature_name == 'colors':
        print(f'\nExtracting {feature_name} [RGB-HSV-Luminance] features...\n')
        extract_colors(vr, output_dir=output_dir, overwrite_ok=overwrite_ok)


    # ----- Extract GIST (global image statistics) features from the video -----
    # GIST is a method used for scene recognition and provides a global
    # representation of an image by considering its structural and textural
    # information.
    # For more details:
    # 1. https://people.csail.mit.edu/torralba/code/spatialenvelope/
    # 2. https://doi.org/10.1016/S0079-6123(06)55002-2
    elif feature_name == 'gist':
        print(f'\nExtracting {feature_name} features...\n')
        # Define parameters for the GIST extraction:
        # `image_size` determines the size of the image (width x height)
        # `nbatches` refers to the number of batches used in the extraction
        # process. To bypass batching and process all images at once, set nbatches to 1.
        params = {
            'image_size': image_size,
            'nbatches': nbatches
        }

        # Call the extract_gist function to perform the GIST feature extraction.
        extract_gist(vr, params, output_dir=output_dir, overwrite_ok=overwrite_ok)


    # ----- Extract motion-energy [spatio-temporal Gabor filters] features -----
    # These features capture the motion information in a video by applying 
    # Gabor filters across space and time.
    # For more details:
    # 1. GitHub repository: https://github.com/gallantlab/pymoten
    # 2. Official documentation: https://gallantlab.org/pymoten/
    elif feature_name == 'moten':
        print(f'\nExtracting {feature_name} [Gabor motion-energy] features...\n')
        # Define parameters for Motion Energy extraction:
        # `image_size`: Size of the image (width x height).
        # `nbatches`: Number of batches in the extraction process. 
        #             Set to 1 to process all images at once.
        # `preprocess_pipeline`: Specifies the preprocessing method.
        #                        'moten' uses the original, slower method,
        #                        while 'opencv' is a faster alternative.
        # `add_noise`: If True, noise will be added to images as part of the preprocessing.
        params = {
            'image_size': image_size,
            'nbatches': nbatches,
            'preprocess_pipeline': inputs['motenprep'],
            'add_noise': True,
        }

        # Call the extract_moten function to perform the motion-energy extraction.
        extract_moten(vr, params, output_dir=output_dir, overwrite_ok=overwrite_ok)

    
    # ----- Detect faces and extract some face-related features using insightface library -----
    # Ref: https://github.com/deepinsight/insightface 
    elif feature_name == 'face_insg':
        face_det_thresh = inputs.get('thresh', None)
        face_det_thresh = face_det_thresh if face_det_thresh else 0.5
        
        print(f'\nExtracting {feature_name} [face-detection] features...\n')
        extract_faces_insight(vr, output_dir=output_dir, overwrite_ok=overwrite_ok,
                              saveviz=saveviz, det_thresh=face_det_thresh)


    # More feature extraction methods can be added here.
    

    # Save presentation time stamp (PTS) values to be used with extracted features 
    if feature_name in features_extract:
        np.save(os.path.join(output_dir,f'{vr.basename}_frame_pts.npy'), vr.pts)


def main(args):
    """
    Main function to run the feature extraction on a given video or all videos in a directory.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    videos_to_process = []

    # Check if a single video file is provided
    if args.video and os.path.isfile(args.video):
        videos_to_process.append(args.video)
    
    # Check if a directory is provided
    elif args.video_dir and os.path.isdir(args.video_dir):
        # Iterate over all files in the directory and add video files to the list
        for filename in os.listdir(args.video_dir):
            if filename.lower().endswith(('.mp4', '.avi', '.mov')):
                videos_to_process.append(os.path.join(args.video_dir, filename))
    
    # Raise an error if neither a valid file nor directory is provided
    else:
        raise SystemExit('Please provide a valid video file or video directory path.')
        
    for video_path in videos_to_process:
        print(f'Processing video: {video_path}')

        # Convert args to a dictionary, e.g., inputs['nbatches'], and update the 'video' entry
        inputs = vars(args)
        inputs['video'] = video_path  # Update the video path

        # Remove None values and update the inputs
        inputs = {k: v for k, v in inputs.items() if v is not None}

        # Call the feature extraction function
        run_feature_extraction(inputs)


if __name__ == "__main__":
    
    # Folder to save extracted features
    output_dir = os.path.abspath('./extracted_features')
    
    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Feature extraction from videos')
    
    # Command-line argument for a single video file
    parser.add_argument('-v', '--video', type=str, #required=True,
                        help='The path to the input video file')
    # Command-line argument for a directory of videos
    parser.add_argument('-d', '--video_dir', type=str, #required=True,
                        help='The path to the directory containing input video files')
    # Define other command-line arguments
    parser.add_argument('--feature', type=str, required=True, choices=features_list,
                        help='The name of the feature to extract')
    parser.add_argument('-o', '--output_dir', type=str, default=output_dir,
                        help='The path to save the output feature file')
    parser.add_argument('--resfac', type=int, default=1, 
                        help='The factor by which to resize the video for feature extraction')
    parser.add_argument('--width', type=int, 
                        help='The new width to resize the video for feature extraction (alternative to resfac)')
    parser.add_argument('--height', type=int, 
                        help='The new height to resize the video for feature extraction (alternative to resfac)')
    parser.add_argument('--nbatches', type=int, default=1,
                        help='The number of batches to split and process the video')
    parser.add_argument('--overwrite', type=str2bool, default=False,
                        help='Whether to overwrite features if they exist in the output directory')
    
    # Additional settings for specific feature extraction scenarios
    parser.add_argument('--motenprep', type=str, default='opencv', 
                        help='moten specific parameter (see the code for details)')
    parser.add_argument('--saveviz', type=str2bool, default=True,
                        help='Whether to save the visualization of detections for semantic feature options')
    parser.add_argument('--thresh', type=float,
                        help='The threshold for detection confidence for semantic feature options')
    
    
    # Parse command-line arguments
    args = parser.parse_args()

    # Run the main function
    main(args)


