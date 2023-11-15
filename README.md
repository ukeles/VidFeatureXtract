# Feature Extraction From Videos

This repository contains code and resources for extracting features from video files.

*Note:* This repository is a work in progress, with many aspects likely to undergo changes or updates.

---
The following features are currently implemented:
- Low-level visual features:
  - [RGB-HSV-Luminance](#rgb-hsv-luminance-features)
  - [GIST](#gist-features)
  - [Motion-energy](#motion-energy-features)
- High-level visual features:
  - [Face detection and basic face-related features](#face-detection-and-basic-face-related-features)
  - ...


## Getting Started

### Base Installation
We recommend using Anaconda and setting up a new environment for this repository. This will ensure that the necessary dependencies are installed and managed in a separate environment, without interfering with other Python projects. Follow these steps to set up the environment on a local machine:

- Downlaod and install Anaconda or Miniconda by following the instructions on the [official Anaconda website](https://www.anaconda.com/).

- Clone or download this repository to your local machine. Open your terminal and execute the following commands:
  ```bash
  git clone https://github.com/ukeles/VidFeatureXtract.git
  cd VidFeatureXtract
  ```

- Use the make_env.yml file provided in the repository to create a new environment that includes the required dependencies:
  ```bash
  conda env create --file make_env.yml
  ```

- Activate the environment and install this repository in editable mode using pip:
  ```bash
  conda activate vidfeats_env
  pip install -e .
  ```
This repository is currently in development. We recommend periodically running the `git pull` command to ensure your local copy stays updated with the latest changes.

This repository utilizes a variety of libraries and pre-trained models for extracting features from video files. To ensure smooth installation and operation, and to avoid conflicts between dependencies, we recommend creating separate conda environments for specific feature extraction tasks. Isolated environments minimize the risk of incompatibility issues with third-party libraries. Detailed instructions for utilizing the various feature extraction functionalities and configuring the additional environments are provided in the usage section.

## Usage
To extract features from a video file or all video files in a directory, use the [`extract_features.py`](extract_features.py) script as follows:
```bash
python extract_features.py --feature <feature_name> [options]
```
Options:
- `-v, --video` : Path to a single video file.
- `-d, --video_dir` : Path to a directory containing video files.
- `--feature` : Name of the feature to extract, see below for specific feature names.
- `-o, --output_dir` : Path to save the output feature files. Default is `./extracted_features`.
- `--resfac` : The factor by which to resize the video for feature extraction. Default is 1.
- `--width` and `--height` : New dimensions to resize the video (alternative to `resfac`).
- `--nbatches` : The number of batches to split and process the video. Default is 1 (process all images at once).
- `--overwrite` : Whether to overwrite features if they already exist in the output directory. Default is False.
- `--motenprep` : `moten` specific parameter for preprocessing.
- `--saveviz` : Whether to save visualizations of detections. Default is True.

Notes: 
- Specifying an output directory with `-o [OUTPUT_DIR]` is optional, but recommended for better organization of features. If not provided, the default output directory is `./extracted_features`.
- For videos with high resolution or extended duration, extracting certain features (e.g., `gist` and `moten`) can be quite memory-intensive. To mitigate excessive memory usage and prevent potential processing bottlenecks, such videos can be split into multiple batches. The batch size can be managed using the `--nbatches` option, the number of batches to split the video.

- To further optimize system resource usage during feature extraction, consider downsizing the frame resolution. This approach can significantly reduce memory demands and, more importantly, decrease the time needed for feature extraction. Frame downsizing can be achieved in two ways: 
  - Use `--resfac` to specify the factor by which the video should be resized for feature extraction. The new dimensions will be calculated as `(new_width, new_height) = (original_width // resfac, original_height // resfac)`.
  - Use `--width` and `--height` to set new dimensions directly, resizing the video to the desired size in pixels.

    Note that in some feature extraction functions, the frames will be resized by the third-party libraries being utilized. In such instances, it is not necessary to explicitly downsize the video frames.


For a full list of options and more details, run:
```bash
python extract_features.py --help
```

### Basic Video Information
To obtain basic information about the video such as resolution, frame count, and frames per second:
```bash
# for a video file
python extract_features.py --feature getinfo -v [VIDEO_PATH]
# or for all video files in a directory
python extract_features.py --feature getinfo -d [VIDEOS_DIR]
```
For example:
```bash
python extract_features.py --feature getinfo -v ./sample_video/video1.mp4 
```

### RGB-HSV-Luminance Features
Color features from videos, including the primary RGB values, the perceptual HSV dimensions, and luminance to indicate brightness, represent basic but informative aspects of visual content. They offer a simple starting point for feature extraction, although they might not be suited for complex models.

To extract basic color features:
```bash
# for a video file
python extract_features.py --feature colors -v [VIDEO_PATH] -o [OUTPUT_DIR]
# or for all video files in a directory
python extract_features.py --feature colors -d [VIDEOS_DIR] -o [OUTPUT_DIR]
```
For example:
```bash
python extract_features.py --feature colors -v ./sample_video/video1.mp4 -o /path/to/output_dir
```

### GIST Features
The [GIST descriptor](https://people.csail.mit.edu/torralba/code/spatialenvelope/) was introduced to summarize the characteristics of a scene within an image. It captures the essence by encoding global structural and textural information, reflecting the spatial layout and the dominant patterns or gradients.

To extract GIST features:
```bash
# for a video file
python extract_features.py --feature gist -v [VIDEO_PATH] -o [OUTPUT_DIR]
# or for all video files in a directory
python extract_features.py --feature gist -d [VIDEOS_DIR] -o [OUTPUT_DIR]
```
Optional for 'gist':
- `--nbatches`: This parameter controls the number of batches used to process the video. By default, it is set to 1, which means the video will be processed in a single batch, including all frames at once. 

For example:
```bash
# use -v for a video file or -d for all video files in a directory
python extract_features.py --feature gist -v ./sample_video/video1.mp4 --resfac 2 --nbatches 2
# or 
python extract_features.py --feature gist -v ./sample_video/video1.mp4 --width 256 --height 256 --nbatches 2 
```

### Motion-Energy Features
[Motion-energy features](https://gallantlab.org/pymoten/) were introduced to capture the dynamic motion information in a video sequence. These features are extracted by using a pyramid of spatio-temporal Gabor filters. 

To extract motion-energy features:
```bash
# use -v for a video file or -d for all video files in a directory
python extract_features.py --feature moten -v [VIDEO_PATH] -o [OUTPUT_DIR]
```
Optional for 'moten':
- `--nbatches`: Number of batches to split and process the video. Default is 1 (process all frames at once).
- `--motenprep`: Choose the preprocessing technique for motion-energy extraction from the video. Options are 'opencv' (faster alternative) or 'moten' (original, slower method). Default is 'opencv'. Both options produce comparable results.

For example:
```bash
# use -v for a video file or -d for all video files in a directory
python extract_features.py --feature moten -v ./sample_video/video1.mp4 --resfac 2 --nbatches 2
# or 
python extract_features.py --feature moten -v ./sample_video/video1.mp4 --width 256 --height 256 --nbatches 2 
```

### Face detection and basic face-related features
This component of feature extraction detects human faces in video frames and computes basic face-related metrics. It generates two types of outputs:
1. Raw Face Detection Data: For each frame, this data includes the locations and dimensions of detected faces (bounding box coordinates [x1, y1, x2, y2]), detection scores, and the coordinates of five facial landmarks for further analysis.
2. Face Feature Matrix: This matrix summarizes three metrics per frame: the number of faces detected, the proportion of the frame's area occupied by faces (cumulative face area), and the average area of the detected faces.

To extract these face-related features:
```bash
# use -v for a video file or -d for all video files in a directory
python extract_features.py --feature face_insg -v [VIDEO_PATH] -o [OUTPUT_DIR]
```
Optional for 'face_insg':
- `--saveviz` : Enable this option to save visualizations of the processed feature detections (e.g., faces) as a video file. The resulting visualization will be stored in the specified `output_dir`. Default is True.

Note that resizing or batching the video is not required for this feature extraction, hence these options are omitted to avoid confusion and usage errors.

For example:
```bash
# use -v for a video file or -d for all video files in a directory
python extract_features.py --feature face_insg -v ./sample_video/video1.mp4
```

Also note that the default threshold value for face detection confidence is set to 0.5; faces with detection scores below this threshold will not be considered. If you encounter too many false positives, consider increasing the threshold value. This can be done as follows:
```bash
python extract_features.py --feature face_insg -v ./sample_video/video1.mp4 --thresh 0.6 
```



### More features are coming here...


## License
This repository is released under the BSD 3-Clause license. See the [LICENSE](LICENSE) file for details.

