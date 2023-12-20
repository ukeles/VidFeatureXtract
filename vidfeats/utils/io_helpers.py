#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Common io helper functions

"""

import os
import argparse

def str2bool(arg_in):
    """
    Convert a string argument to a boolean.
    
    Parameters:
    -----------
    arg_in : str or bool
        Input argument to be converted.

    Returns:
    --------
    bool
        True or False based on the input string.

    Raises:
    -------
    argparse.ArgumentTypeError
        If the provided argument cannot be converted to a boolean.
    """
    if isinstance(arg_in, bool):
        return arg_in
    elif arg_in.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif arg_in.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean [True or False] value expected.')



def check_outputfile(output_file, overwrite_ok):
    """
    Check the existence of the specified output file and optionally create its directory.
    
    This function raises an exception if the output file exists and overwriting is not allowed.
    If the directory containing the output file doesn't exist, it creates the directory.
    
    Parameters
    ----------
    output_file : str
        Path to the output file to be checked.
    overwrite_ok : bool
        If True, allows overwriting of an existing file. 
        If False, raises an error when the file exists.
        
    Raises
    ------
    SystemExit
        If the output file already exists and `overwrite_ok` is set to False.
        
    Examples
    --------
    >>> check_outputfile("/path/to/output/file.npy", False)
    SystemExit: Overwriting feature extraction file:
    /path/to/output/file.npy
    set: overwrite=True to overwrite!
    
    >>> check_outputfile("/path/to/output/file.npy", True)
    # No exception raised, and directory "/path/to/output" is created if it doesn't exist.
    """
    
    if os.path.isfile(output_file) and not overwrite_ok:
        raise SystemExit(f"Overwriting feature extraction file:\n{output_file}"+
                         "\nset: overwrite=True to overwrite!")

    # Create the directory for the output file if it doesn't exist.
    os.makedirs(os.path.dirname(output_file), exist_ok=True)


def finalize_and_save(output_vid_file, outfile_pkl, feats_dict, 
                      outfile_arr=None, feats_arr=None):
    """
    Finalizes video writing if applicable and saves feature data to files.

    Parameters
    ----------
    output_vid_file : cv2.VideoWriter
        The VideoWriter object used for saving the visualization video.
    outfile_pkl : str
        File path for the pickle file to save feature data.
    feats_dict : dict
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
        pickle.dump(feats_dict, pfile)
    
    if outfile_arr is not None and feats_arr is not None:
        np.save(outfile_arr, feats_arr)
        

import base64
from io import BytesIO
import numpy as np
from PIL import Image

def array2str_encoder(arr_in):
    """
    Encode a 2D array representing a body parts mask into a compressed string format.

    Parameters
    ----------
    arr_in : np.ndarray
        A 2D numpy array representing the body parts mask.

    Returns
    -------
    str
        The base64 encoded string of the mask image.
    """
    # Ensure that converting the array to uint8 is a safe operation.
    assert arr_in.max() < np.iinfo(np.uint8).max
    assert arr_in.min() >= np.iinfo(np.uint8).min
    
    # Convert the array to uint8, a standard format for images
    img_dum = arr_in.astype(np.uint8)
    # Create an image object from the array
    im_out = Image.fromarray(img_dum)
    # Create an in-memory byte-stream for the image
    fstream = BytesIO()
    # Save the image to the stream in PNG format for efficient compression
    im_out.save(fstream, format="png", optimize=True)
    # Encode the image to base64 and decode to a string
    s_out = base64.encodebytes(fstream.getvalue()).decode()
    return s_out


def str2array_decoder(s_in):
    """
    Decode a base64 encoded string back into a 2D array.

    Parameters
    ----------
    s_in : str
        The base64 encoded string of the image.

    Returns
    -------
    np.ndarray
        The decoded image as a 2D numpy array.

    Notes
    -----
    The function assumes the string is encoded from an image in PNG format.
    The returned array's datatype and shape depend on the original image.
    """
    # Convert the base64 string back to a byte stream
    fstream = BytesIO(base64.decodebytes(s_in.encode()))
    # Open the image from the byte stream and convert to a numpy array
    return np.asarray(Image.open(fstream))


