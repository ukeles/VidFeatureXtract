#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""


def compute_downsample_size(down_size, resolution_wh):
    """
    Compute the downsample size based on the given input size and original resolution.

    This function determines the downsample size based on different formats of `down_size`:
    1. If `down_size` is None, it uses the original resolution.
    2. If `down_size` is an integer or float, it divides the original resolution by this number.
    3. If `down_size` is a tuple or list of two values, it uses these values directly.

    Parameters
    ----------
    down_size : None, int, float, tuple or list
        The desired downsample size. 
        If None, the original resolution is used.
        If int or float, the original resolution is divided by this number.
        If tuple or list, it should contain two values for width and height.
    resolution_wh : tuple
        The original resolution (width, height).

    Returns
    -------
    tuple
        The computed downsample size (width, height).

    Raises
    ------
    NotImplementedError
        If `down_size` type is not None, int, float, tuple or list of two values.

    Examples
    --------
    >>> compute_downsample_size(2, (1280, 720))
    (640, 360)

    >>> compute_downsample_size(None, (1280, 720))
    (1280, 720)

    >>> compute_downsample_size((800, 600), (1280, 720))
    (800, 600)

    """
    
    if down_size is None:
        down_use = resolution_wh 
    elif isinstance(down_size, (int, float)):
        down_use = (resolution_wh[0] // down_size, resolution_wh[1] // down_size)
    elif isinstance(down_size, (tuple, list)):
        assert len(down_size) == 2
        down_use = down_size
    else:
        raise NotImplementedError(f"Undefined type for down_size:{down_size}."+
                                  " Should be a single number, or tuple/list of two values!")

    return down_use


