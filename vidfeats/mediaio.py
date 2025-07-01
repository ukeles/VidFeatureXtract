#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

ToDo: Image and Audio classes for image and audio only stimulus files.

"""

import os
import numpy as np
import av

def get_pts_av(video_file, fps):
    """
    Retrieves the presentation timestamp (PTS), decoding timestamp (DTS), 
    and frame count from a video file.

    Parameters
    ----------
    video_file : str
        The path to the input video file.
    fps : float
        Frames per second of the video.
    Returns
    -------
    np.ndarray
        Array of presentation timestamps (PTS) in seconds.
    np.ndarray
        Array of decoding timestamps (DTS) in seconds.
    int
        Total number of frames in the video.
        
    Notes: 
        In some stim videos pts obtained via PyAv/Pims was not monotonically increasing, 
        but oscillates. Comparing the dts obtained from PyAv and the pts obtained from  
        ffpyplayer.player.MediaPlayer, I decided to use the dts as pts for this problematic case.
        In general, if the video file does not have any such issues, pts and dts values are equal.
    """
    print('Retrieving the presentation timestamps (PTS)...', flush=True)

    with av.open(video_file) as container:
        # note that this assumption was used in PyAVReaderIndexed() too.
        stream = container.streams.video[0] 

        # note that getting frames from container.decode(stream) is more straightforward 
        # than using packets from container.demux(stream) to handle dts values. 
        dts, pts = [], []
        for frame in container.decode(stream): 
            dts_time = float(frame.dts*frame.time_base) if frame.dts is not None else np.nan
            dts.append(dts_time)
            pts.append(frame.time) # alternatively: float(frame.pts * frame.time_base)

    pts = np.asarray(pts)
    dts = np.asarray(dts)
    # dts_raw = dts.copy()

    if np.isnan(dts).any():
        nan_mask = np.isnan(dts)
        first_nan_index = nan_mask.argmax()
    
        # Ensure NaNs are only at the end
        assert np.all(np.isnan(dts[first_nan_index:]))
    
        # Ensure PTS does not contain NaNs
        assert not np.isnan(pts).any()
    
        # Check if PTS is monotonically increasing and 
        # use its values to fill DTS
        if np.all(pts[:-1] < pts[1:]):
            dts[nan_mask] = pts[nan_mask]
            assert np.all(dts[:-1] < dts[1:])
        else: # otherwise fill by using the avg_frame_time
            # Calculate the number of NaNs to fill
            num_nans = len(dts) - first_nan_index
            avg_frame_time = 1.0 / fps # in seconds

            # Fill NaNs by extending the linear trend
            dts[first_nan_index:] = dts[first_nan_index-1] + avg_frame_time*np.arange(1,num_nans+1)

            # Ensure DTS does not contain any other NaNs
            assert not np.isnan(dts).any()
            assert np.all(dts[:-1] < dts[1:])

    return pts, dts, len(pts)



# this option is ~4x slower and requires the package 'ffpyplayer', but quite reliable.
import time
def get_pts_mediaplayer(video_file, fps):
    """
    Extracts presentation timestamps (PTS) and frame count from a video file
    using the MediaPlayer from the ffpyplayer library.

    Parameters
    ----------
    video_file : str
        The path to the input video file.
    fps : float
        Frames per second of the video.

    Returns
    -------
    np.ndarray
        Array of presentation timestamps (PTS) with the last frame timestamp
        extrapolated based on the average frame time.
    int
        Total number of frames in the video.
    """
    print('Running MediaPlayer...')

    from ffpyplayer.player import MediaPlayer

    # MediaPlayer options
    ff_opts = {
        'out_fmt': 'rgb24',
        'fast': True,
        'an': True,  # Ignore audio stream
        'framedrop': False, # Must be enabled to process all frames
        'sync': 'video'
    }

    # Initialize MediaPlayer
    player = MediaPlayer(video_file, ff_opts=ff_opts)

    pts_med = []
    while True:
        frame, val = player.get_frame()
        if val == 'eof':
            break
        elif frame is None:
            time.sleep(0.01)
        else:
            _, t = frame
            pts_med.append(t)

    # Clean up MediaPlayer
    player.close_player()

    # Assertions to check PTS consistency
    assert pts_med[0] == pts_med[1], "First two PTS values are not equal."
    assert not np.isnan(pts_med).any(), "PTS contains NaN values."

    # Prepare the PTS array with an extra NaN value for the last frame
    pts_med_r = np.r_[pts_med[1:], np.nan]

    # Calculate average frame time and fill the last NaN value
    avg_frame_time = 1. / fps
    pts_med_r[-1] = pts_med_r[-2] + avg_frame_time

    # Ensure PTS is monotonically increasing
    assert np.all(pts_med_r[:-1] < pts_med_r[1:]), "PTS is not monotonically increasing."

    return pts_med_r, len(pts_med)



def select_indices_centered(k, n):
    """
    Select 'k' evenly spaced indices from a sequence of length 'n'.
    Adds an offset to make the selection more centered within each interval.
    
    Parameters
    ----------
    k : int
        The number of elements to select.
    n : int
        The length of the sequence from which elements are to be selected.

    Returns
    -------
    list of int
        A list of 'k' indices
    """
    assert k>0 and n>0
    
    return [ int(ii * n / k) + int(n / (2 * k)) for ii in range(k) ]
    # return [ ii * n // k + n // (2 * k) for ii in range(k) ]


video_readers = ['pims_av']

class Video:
    """
    Video class provides an interface for loading a video file and 
    extracting frames from the video.
    
    Notes:
        1- Some options for 'reader' are: pims, decord, opencv, mmcv.
           Currently: pims.PyAVReaderIndexed() is used to get frames and 
           pims.PyAVReaderTimed() to get video meta information.
        
    """

    def __init__(self, file, reader='pims_av', quick_info=False):
        """
        Initializes the Video class.
        
        Parameters:
        - file: Path to the video file.
        - reader: String representing the method to read the video. 
                  Default is 'pims_av'.
        """
        
        if not os.path.isfile(file):
            raise FileNotFoundError(f'No such video file: {file}')
        
        self.file = file
        self.basefile = os.path.basename(file)
        self.basename = os.path.splitext(self.basefile)[0]
        self.quick_info = quick_info
        
        self.reader = reader
        assert reader in video_readers, f"""reader={reader} is not implemented. Options are:
            {video_readers} """
            
        self.get_metadata()


    def get_metadata(self):
        """
        Reads and loads various meta/codec data from the video file.
        
        Returns:
        -------
        None.
        """
        if self.reader == 'pims_av':
            from pims import PyAVReaderTimed, PyAVReaderIndexed

            # Initial load to get general metadata for the video
            vr = PyAVReaderTimed(self.file)
                
            # Extracting various video metadata properties
            self.duration = vr.duration
            self.fps = vr.frame_rate
            self._frame_count_timed = len(vr) 
            self.width = vr.frame_shape[1]
            self.height = vr.frame_shape[0]
            self.resolution = (self.width, self.height)
            self.pts_quick = None
            
            if self.quick_info:
                
                pts_fname = os.path.splitext(self.file)[0] + '_pts.npy'
                if os.path.isfile(pts_fname):
                    self.pts_quick = np.load(pts_fname)
                    
                return 
            
            # Second load to get more accurate frame count of the video
            print('Counting frames with pims.PyAVReaderIndexed()...', flush=True)
            vr = PyAVReaderIndexed(self.file)
            self.frame_count = len(vr)
            self.obj = vr
            
            # Calculate presentation time stamp (PTS) values in seconds
            # read PTS and DTS values from the video file
            pts_arr, dts_arr, nframes_chk = get_pts_av(self.file, self.fps)
            assert self.frame_count == nframes_chk
            
            # these should be equal for packets of length 1 | this assert can be removed. 
            # toc_lens, toc_ts = np.asarray(vr.toc['lengths']), np.asarray(vr.toc['ts'])
            # toc_pts = toc_ts * float(vr._time_base)
            # assert np.allclose(pts_arr[:len(toc_lens)], toc_pts[:len(toc_lens)])
            
            if np.all(pts_arr[:-1] < pts_arr[1:]):
                assert all(pts_arr >= 0), 'Problem in getting PTS values, you may try get_pts_mediaplayer()'
                self.pts = pts_arr
                self.pts_org = None
                self.dts_org = dts_arr
            else:
                assert all(dts_arr >= 0), 'Problem in getting DTS values, you may try get_pts_mediaplayer()'
                self.pts = dts_arr
                self.pts_org = pts_arr
                self.dts_org = None
            
            # These are used if extraction_fps is provided in inputs.
            self.extraction_fps = None
            self.extraction_pts = None
            self.extraction_frames = None
            
        # [Other video readers can be implemented here]


        # Warning if frame counts from different methods don't match
        if self._frame_count_timed != self.frame_count:
            print("The number of frames obtained from metadata and manual counting are different!\n"+
                  f"> Difference (Indexed - Timed) [{self.frame_count} - {self._frame_count_timed}]: {self.frame_count-self._frame_count_timed} "+
                  f"\nThis code takes the frame count as {self.frame_count}.")


    def __getitem__(self, indx):
        """
        Returns a single video frame or a sequence of frames from 
        the video object at the given index.
        """
        if self.reader == 'pims_av':
            return self.obj[indx]
        # [More conditions for different types of video readers can be added here]


    def get_frame(self, indx):
        """
        Returns a single frame from the video object at the given index.
        """
        if self.reader == 'pims_av':
            return self.obj[indx]
        # [Other readers can be implemented here]
        
        
    def fps_resample_basic(self, extraction_fps):
        """
        Get video frames at a rate of 'extraction_fps' 
        
        Parameters:
        -----------
        extraction_fps: int
            desired frame rate for extraction

        Returns
        -------
        fps_splits_down: np.dnarray
            1st column: dummy chunk indicies, can be used to re-split the array,
            2nd column: resampled frame numbers, refering to the original frame numbers,
            3rd column: resampled PTS values.  
        """
        
        nframes = self.frame_count  # Number of frames in the video.
        vr_pts = self.pts
        
        chunks = ( vr_pts / 1. ).astype(int) + 1 # "/ 1." is for an initial splitting into 1 sec blocks
        fps_dum = np.c_[ chunks, np.arange(nframes), vr_pts ]
        fps_splits = np.split(fps_dum, np.where(np.diff(chunks))[0]+1)
        
        fps_splits_down = []
        for sp_cnt, sp_ii in enumerate(fps_splits):
            
            if extraction_fps < len(sp_ii):
                use_inds = select_indices_centered( extraction_fps, len(sp_ii) )
                fps_splits_down.append( sp_ii[use_inds] )
            else:
                fps_splits_down.append( sp_ii )       

        fps_splits_stack = np.vstack(fps_splits_down)
        
        self.extraction_fps = extraction_fps
        self.extraction_frames = fps_splits_stack[:, 1].astype(int)
        self.extraction_pts = fps_splits_stack[:, 2]
        
        return fps_splits_down


    def examine_nframes(self):
        """
        Count the total number of frames in the video file using alternative methods.
        If the frame rate is not constant across the video, different counting methods
        might yield different results. Manual counts (e.g., pims.PyAVReaderIndexed) seem 
        to be the most accurate for feature extraction purposes.

        Returns:
        -------
        Prints frame counts obtained using different methods.
        """

        print('\n\nExamines the number of frames using different methods\n')

        print(f'Frame count with pims_av: {self.frame_count} '+
              f'with fps: {self.fps} --- _frame_count_timed is {self._frame_count_timed}\n')
        
        # Counting frames using OpenCV
        try:
            import cv2
            cap = cv2.VideoCapture(self.file)

            if not cap.isOpened():
                print("Error opening video file")
                
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f'Frame count with openCV.CAP_PROP_FRAME_COUNT: {nframes} with fps: {fps}')
            
            # Manual counting of frames
            cnt_ii = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cnt_ii += 1
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            print(f'Total number of frames with openCV (manual count): {cnt_ii}\n')
            
            cap.release()
            cv2.destroyAllWindows()

        except ModuleNotFoundError:
            print('openCV module is not installed!')

        # Counting frames using decord
        try:
            from decord import VideoReader, cpu
    
            vid = VideoReader(self.file, ctx=cpu(0))
            nframes = len(vid)
            fps = vid.get_avg_fps()
            print(f'Frame count with decord library: {nframes} with fps: {fps}\n')

        except ModuleNotFoundError:
            print('decord module is not installed!')

        # Counting frames using mmcv
        try:
            import mmcv
            vid = mmcv.VideoReader(self.file)

            mmcv_cnt = 0
            for _, fr in enumerate(vid):  # vid is iterable
                if fr is not None:
                    mmcv_cnt += 1

            print(f'Frame count with mmcv library: {mmcv_cnt} '+
                  f'with fps: {vid.fps} --- metadata info gives {len(vid)}\n')

        except ModuleNotFoundError:
            print('mmcv module is not installed!')

