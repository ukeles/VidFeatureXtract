#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

ToDo: Image and Audio classes for image and audio only stimulus files.

"""

import os
import numpy as np

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

    def __init__(self, file, reader='pims_av'):
        """
        Initializes the Video class.
        
        Parameters:
        - file: Path to the video file.
        - reader: String representing the method to read the video. 
                  Default is 'pims_av'.
        """
        self.file = file
        self.basefile = os.path.basename(file)
        self.basename = os.path.splitext(self.basefile)[0]
        
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
            
            # Second load to get more accurate frame count of the video
            print('Counting frames with pims.PyAVReaderIndexed()...')
            vr = PyAVReaderIndexed(self.file)
            self.frame_count = len(vr)
            self.obj = vr
            
            # Calculate presentation time stamp (PTS) values in seconds
            toc_lens, toc_ts = np.asarray(vr.toc['lengths']), np.asarray(vr.toc['ts'])
            tbase = float(vr._time_base)
            
            unq_toc_lens = np.unique(toc_lens)
            if len(unq_toc_lens) == 1:
                assert unq_toc_lens[0] == 1
            # special cases for videos with variable frame rate (VFR)
            elif unq_toc_lens.tolist() == [1, 2]:
                assert np.sum(toc_lens == 2) == 1
                assert toc_lens[-1] == 2
                dtoc_ts = np.diff(toc_ts)
                # usually the rate oscillates, thus we use dtoc_ts[-2] instead of dtoc_ts[-1]
                toc_ts = np.r_[toc_ts[0], dtoc_ts, dtoc_ts[-2] ].cumsum()
            else:
                raise NotImplementedError(f'This case is not expected!\n{unq_toc_lens.tolist()}')
            
            self.pts = toc_ts * tbase
            # These are used if extraction_fps is provided in inputs.
            self.extraction_fps = None
            self.extraction_pts = None
            self.extraction_frames = None
            
            
        # [Other video readers can be implemented here]


        # Warning if frame counts from different methods don't match
        if self._frame_count_timed != self.frame_count:
            print("The number of frames obtained from metadata and manual counting are different!\n"+
                  f"difference (Indexed-Timed): {self.frame_count-self._frame_count_timed} "+
                  "\n--might need to confirm the actual number from the experiment log files. "+
                  "This might be because the video has varying frame rates throughout its duration, "+
                  " i.e., variable frame rate.\n"+
                  f"This code will take the frame count as {self.frame_count}.")

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


