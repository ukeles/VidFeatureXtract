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
  - [Human body part detection using DensePose](#detection-and-segmentation-of-human-body-parts)
  - [Semantic segmentation using OneFormer](#semantic-segmentation)


## Getting Started

### Installation
We recommend using Anaconda and setting up a new environment for this repository. This will ensure that the necessary dependencies are installed and managed in a separate environment, without interfering with other Python projects. To use this repo for the extraction of low-level visual features or detecting human faces in video frames, follow the base installation steps. For more advanced functionalities such as human body parts detection and semantic segmentation, please follow the advanced installation section provided below.

#### Base Installation
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
  conda activate vidfeats_base
  pip install -e .
  ```
This repository is currently in development. We recommend periodically running the `git pull` command to ensure your local copy stays updated with the latest changes.


#### Advanced Installation
Create a new conda virtual environment and activate it:
```bash
conda create -n vidfeats_seg
conda activate vidfeats_seg
```

Install python and pip
```bash
conda install -c conda-forge python=3.11 pip setuptools wheel
```

It is recommended to first install the Cuda Toolkit for optimal compatibility and ease of installation for other libraries that rely on the `CUDA_HOME` environment variable. After installing the Cuda Toolkit, PyTorch should be installed next, followed by any remaining packages, to ensure an efficient and smooth setup.

See https://pytorch.org/get-started/locally/ for available versions of Cuda Platform used for PyTorch.
```bash
# use either
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
# or | ensure that cudatoolkit version is consistent with pytorch-cuda version below
conda install -c conda-forge cudatoolkit cudatoolkit-dev

# select either (i) or (ii) to set the environment variable CUDA_HOME
# echo $CUDA_HOME might be used the check if it is already set.
# (i) if installed via conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
export CUDA_HOME=$CONDA_PREFIX
# (ii) if installed via conda install -c conda-forge cudatoolkit cudatoolkit-dev
export CUDA_HOME=$CONDA_PREFIX/pkgs/cuda-toolkit

# then install PyTorch components
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Remaining dependicies of `vidfeats` repo can be installed using `pip`:
```bash
git clone https://github.com/ukeles/VidFeatureXtract.git
cd VidFeatureXtract
python -m pip install -r requirements.txt 
python -m pip install -r requirements_seg.txt
# finally install this repository in editable mode using pip:
python -m pip install -e .
```

In order to use [DensePose](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose) model for detecting and segmenting human body parts, please install the [Detectron2](https://github.com/facebookresearch/detectron2) library and the DensePose module. 
```bash
# To install the detectron2 library:
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# to install the DensePose library for body parts detection:
# see: https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/doc/GETTING_STARTED.md
python -m pip install 'git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose'
```
For additional information about Detectron2, see the [official documentation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).


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
# for a video file
python extract_features.py --feature getinfo -v ./sample_video/video1.mp4
# or for all video files in a directory
python extract_features.py --feature getinfo -d ./sample_video
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

### Detection and Segmentation of Human Body Parts
We leverage the [DensePose](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose) module from [Detectron2](https://github.com/facebookresearch/detectron2) for detecting and segmenting human body parts within video frames. DensePose is instrumental in extracting detailed human body shapes and poses. It achieves this by mapping all human pixels in an RGB image to the 3D surface of the human body. We specifically utilize DensePose for frame-by-frame analysis, enabling us to map human pixels into three distinct body part areas: the head, hands, and other body parts. The extracted data is then saved in a format that is efficient in terms of memory usage.

To perform human body part detection and segmentation:
```bash
# use -v for a video file or -d for all video files in a directory
python extract_features.py --feature densepose -v [VIDEO_PATH] -o [OUTPUT_DIR]
```

### Semantic Segmentation
We utilize the OneFormer framework for a detailed pixel-wise segmentation of video frames. OneFormer segments and classifies various objects and elements in a scene, delivering high accuracy and providing rich contextual information. For details on OneFormer and its capabilities, please refer to their [official documentation](https://github.com/SHI-Labs/OneFormer). We employ the OneFormer model via the HuggingFace [transformers](https://huggingface.co/docs/transformers/main/en/model_doc/oneformer) library. 


To perform semantic segmentation with OneFormer:
```bash
# use -v for a video file or -d for all video files in a directory
# option 1: to employ OneFormer model trained on the ADE20k dataset (150 categories, large-sized version, Swin backbone)
# see: https://huggingface.co/shi-labs/oneformer_ade20k_swin_large
python extract_features.py --feature oneformer_ade -v [VIDEO_PATH] -o [OUTPUT_DIR]
# option 2: to employ OneFormer model trained on the COCO dataset (133 categories, large-sized version, Swin backbone):
python extract_features.py --feature oneformer_coco -v [VIDEO_PATH] -o [OUTPUT_DIR]
```

Optional for 'oneformer_ade' and 'oneformer_coco':
- `--extraction_fps`: The frame rate to sample the video for feature extraction. Use this to specify a lower frame rate for processing, by which we can control the number of frames per second to be analyzed, enabling a balance between processing speed and the level of detail captured from the video. 
- `--saveviz`: Enable this option to save visualizations of the processed feature detections (e.g., faces) as a video file. The resulting visualization will be stored in the specified output_dir. Default is True.

Note that resizing or batching the video is not required for this feature extraction, hence these options are omitted to avoid confusion and usage errors.

### More features are coming here...

## License
This repository is released under the BSD 3-Clause license. See the [LICENSE](LICENSE) file for details.

