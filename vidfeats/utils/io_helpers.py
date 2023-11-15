#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

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
        If True, allows overwriting of an existing file. If False, raises an error when the file exists.
        
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


