#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Adapted from:
https://github.com/imoken1122/GIST-feature-extractor/
'''

import numpy as np
import numpy.matlib as nm
import numpy.fft as f

class LMgist:
    """
    Class to extract GIST features from images.
    """

    def __init__(self, param, img_size_wh=None):
        """
        Initialize LMgist object.

        Parameters:
        - param: Dictionary containing GIST extraction parameters.
        - img_size_wh: Tuple specifying image width and height (width, height).
        """
        self.param = param

        # If image dimensions provided, create Gabor filters.
        if img_size_wh is None:
            self.param["G"] = None
        else:
            self.param["G"] = self._createGabor(self.param["orientationsPerScale"],\
                        np.array(img_size_wh[::-1])+2*self.param["boundaryExtension"])

    def _createGabor(self, orr, n):
        """
        Create Gabor filters based on orientations and image dimensions.

        Parameters:
        - orr: List of orientations.
        - n: Image dimensions.

        Returns:
        - G: Gabor filters.
        """
        gabor_param = []
        Nscalse = len(orr)
        Nfilters = sum(orr)

        # Ensure n has two elements, one for each image dimension.
        if len(n) == 1:
            n = [n[0], n[0]]

        for i in range(Nscalse):
            for j in range(orr[i]):
                gabor_param.append([0.35, 0.3 / (1.85 ** i), 16 * orr[i] **2 / 32**2, np.pi / orr[i] * j])

        gabor_param = np.array(gabor_param)
        fx, fy = np.meshgrid(np.arange(-n[1] / 2, n[1] / 2), np.arange(-n[0] / 2, n[0] / 2))
        fr = f.fftshift(np.sqrt(fx ** 2 + fy ** 2))
        t = f.fftshift(np.angle(fx + 1j * fy))

        G = np.zeros((n[0], n[1], Nfilters))
        for i in range(Nfilters):
            tr = t + gabor_param[i, 3]
            tr += 2 * np.pi * (tr < -np.pi) - 2 * np.pi * (tr > np.pi)
            G[:, :, i] = np.exp(-10 * gabor_param[i, 0] * (fr/ n[1] / gabor_param[i, 1] - 1)**2 -
                                2 * gabor_param[i, 2] * np.pi * tr**2)

        return G

    def _more_config(self, img):
        """Configuration check and setup for GIST extraction."""
        
        if img.ndim != 2:
            raise ValueError('Input image should be gray-scale!')

        if self.param.get('imageSize') is not None:
            raise SystemExit('Perform image resizing externally!')
        
        if self.param["G"] is None:
            self.param["G"] = self._createGabor(self.param["orientationsPerScale"],\
                            np.array(img.shape[:2])+2*self.param["boundaryExtension"])

    def _preprocess(self, img):
        """Preprocess the image by normalizing its values to the range [0, 255]."""
        img = (img - np.min(img))
        if np.sum(img) != 0:
            img = 255 * (img / np.max(img))
        return img

    def _prefilt(self, img):
        """Pre-filter the image."""
        w = 5
        fc = self.param["fc_prefilt"]
        s1 = fc / np.sqrt(np.log(2))
        img = np.log(img + 1)
        img = np.pad(img, [w, w], "symmetric")
        sn, sm = img.shape
        n = max(sn, sm)
        n += n % 2

        # Adjust image dimensions using padding.
        if sn == sm:
            img = np.pad(img, [0, n - sn], "symmetric")
        elif sn < sm:
            img = np.pad(img, [0, n - sn], "symmetric")[:, :sm + (1 if sm % 2 != 0 else 0)]
        else:
            img = np.pad(img, [0, n - sm], "symmetric")[sn - sm:]

        fx, fy = np.meshgrid(np.arange(-n/2, n/2), np.arange(-n/2, n/2))
        gf = f.fftshift(np.exp(-(fx**2 + fy**2) / (s1**2)))
        output = img - np.real(f.ifft2(f.fft2(img) * gf))
        localstd = np.sqrt(abs(f.ifft2(f.fft2(output ** 2) * gf)))
        output = output / (0.2 + localstd)

        # Remove padding after processing
        return output[w:sn-w, w:sm-w]

    def _gistGabor(self, img):
        """
        Extract GIST features using Gabor filters.

        Parameters:
        - img: Pre-filtered image.

        Returns:
        - g: GIST features.
        """
        w = self.param["numberBlocks"]
        G = self.param["G"]
        be = self.param["boundaryExtension"]
        ny, nx, Nfilters = G.shape
        W = w[0] * w[1]
        N = 1
        g = np.zeros((W * Nfilters, N))

        # Apply symmetric padding
        img = np.pad(img, [be, be], "symmetric")
        img = f.fft2(img)

        k = 0
        for n in range(Nfilters):
            ig = abs(f.ifft2(img * nm.repmat(G[:, :, n], 1, 1)))
            ig = ig[be:ny-be, be:nx-be]
            v = self._downN(ig, w)
            g[k:k+W, 0] = v.reshape([W, N], order="F").ravel()
            k += W
        return np.array(g)

    def _downN(self, x, N):
        """
        Downsample the image by averaging over blocks.

        Parameters:
        - x: Image to downsample.
        - N: Downsample dimensions.

        Returns:
        - y: Downsampled image.
        """
        nx = list(map(int, np.floor(np.linspace(0, x.shape[0], N[0]+1))))
        ny = list(map(int, np.floor(np.linspace(0, x.shape[1], N[1]+1))))
        y = np.zeros((N[0], N[1]))

        for xx in range(N[0]):
            for yy in range(N[1]):
                a = x[nx[xx]:nx[xx+1], ny[yy]:ny[yy+1]]
                v = np.mean(a)
                y[xx, yy] = v

        return y

    def gist_extract(self, img):
        """
        Extract GIST features from the image.

        Parameters:
        - img: Input grayscale image.

        Returns:
        - GIST features.
        """
        self._more_config(img)   # Check and setup configurations
        img = self._preprocess(img)   # Normalize the image
        output = self._prefilt(img)   # Pre-filter the image
        gist = self._gistGabor(output)   # Extract GIST features using Gabor filters

        return gist.ravel()
