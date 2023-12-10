#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Human body parts detection and exctraction of some related features from videos

"""

import os
import cv2
import numpy as np
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

from densepose import add_densepose_config
from densepose.structures import DensePoseChartPredictorOutput, DensePoseEmbeddingPredictorOutput
from densepose.vis.extractor import DensePoseOutputsExtractor, DensePoseResultExtractor

from ..utils.io_helpers import check_outputfile, array2str_encoder, finalize_and_save
from .facebody_helpers import densepose_results2bodypart_layers
from .facebody_helpers import densepose_facehand_layers, visualize_bpartdetections


# keep here so that we won't need to import densepose helper functions in facebody_helper.py
def get_densepose_result(pred_in):
    """
    Processes DensePose prediction results and prepares a structured output.

    Parameters
    ----------
    pred_in : DensePosePredictorOutput
        The prediction output from a DensePose model.

    Returns
    -------
    dict or None
        A dictionary containing processed DensePose results, including bounding boxes 
        (in XYXY format), scores, and encoded parts. Returns None if no detection is made.
    """
    result_prep = {}
    # Check if predictions contain bounding boxes
    if pred_in.has("pred_boxes"):
        pred_boxes_XYXY = pred_in.get("pred_boxes").tensor.numpy()
        result_prep["dp_boxes_xyxy"] = np.round(pred_boxes_XYXY).astype(int)
        result_prep["dp_scores"] = pred_in.get("scores").numpy()
        
        # Check and process DensePose predictions
        if pred_in.has("pred_densepose"):
            # Choose the correct extractor based on the type of DensePose output
            if isinstance(pred_in.pred_densepose, DensePoseChartPredictorOutput):
                extractor = DensePoseResultExtractor()
            elif isinstance(pred_in.pred_densepose, DensePoseEmbeddingPredictorOutput):
                extractor = DensePoseOutputsExtractor()

            densepose_result = extractor(pred_in)[0]
            
            # Encode each part of the DensePose result
            this_frame_densepose_str = [array2str_encoder(dii.labels.numpy()) for dii in densepose_result]
            result_prep["dp_parts_str"] = this_frame_densepose_str

            return result_prep

    # Return None if there are no detections
    return None


def extract_bodyparts_densepose(vr, output_dir, modelzoo_dir, overwrite_ok=False, 
                                saveviz=True, det_thresh=0.5):
    """
    Extracts human body part areas from a video using the DensePose model in Detectron2, 
    saves extracted data, and optionally visualizes results in an output video.

    Parameters
    ----------
    vr : VideoReader
        A VideoReader object providing access to video frames.
    output_dir : str
        The directory where output files will be saved.
    modelzoo_dir: str
        The directory containing pre-trained model for densepose.
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

    nframes = vr.frame_count  # Number of frames in the video.
    frame_width, frame_height = vr.resolution  # Resolution of the video (width, height).
    vid_fps = vr.fps # Frames per second of the video.
    vid_basename = f'{vr.basename}_densepose_thresh{det_thresh}'
    
    # Output file paths
    outfile_pkl = os.path.join(output_dir, f'{vid_basename}.pkl')
    
    # Check if output files already exist
    check_outputfile(outfile_pkl, overwrite_ok)
    
    # ------- Densepose settings: -------
    # Download pre-trained models from:
    # https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/doc/DENSEPOSE_IUV.md#ModelZoo
    # Original from Guler et al. 2018
    config_fpath = os.path.join(modelzoo_dir,'densepose_rcnn_R_101_FPN_s1x_legacy.yaml')
    model_fpath  = os.path.join(modelzoo_dir,'model_final_ad63b5.pkl')

    # config_fpath = os.path.join(modelzoo_dir,'densepose_rcnn_R_101_FPN_DL_s1x.yaml')
    # model_fpath  = os.path.join(modelzoo_dir,'model_final_844d15.pkl')

    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(config_fpath)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = det_thresh
    cfg.MODEL.WEIGHTS = model_fpath
    
    predictor = DefaultPredictor(cfg)
    # ------- o -------
    
    output_vid_file = None
    if saveviz:
        output_fname = os.path.join(output_dir, f'{vid_basename}.mp4')
        output_vid_file = cv2.VideoWriter(filename=output_fname,
                                          fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                                          fps=float(vid_fps),
                                          frameSize=(frame_width, frame_height),
                                          isColor=True)
    
    feats_data = {}
    for fii, frame_ii in enumerate(tqdm(vr, total=nframes)):
        # Convert image format from RGB to BGR for detectron/densepose
        img = cv2.cvtColor(frame_ii, cv2.COLOR_RGB2BGR)
        
        model_output = predictor(img)
        predictions = model_output["instances"].to("cpu")
        result_thisframe = get_densepose_result(predictions)
    
        if result_thisframe is not None:
            feats_data[f'frame_{fii+1}'] = result_thisframe
    
            if saveviz:
                img_layers, _ = densepose_results2bodypart_layers(result_thisframe, img.shape)
                if img_layers is not None:
                    img_layers_red = densepose_facehand_layers(img_layers)
                    img = visualize_bpartdetections(img, img_layers_red)
    
        # Visualization part
        if saveviz:
            output_vid_file.write(img)
    
    # Finalize video writing and save feature data
    finalize_and_save(output_vid_file, outfile_pkl, feats_data)
    
