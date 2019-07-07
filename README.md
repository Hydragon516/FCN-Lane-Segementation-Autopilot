# About

Implement simple lane division using FCN(Fully Convolutional Network). The FCN model referred to "https://github.com/mvirgo/MLND-Capstone".

# Requirements

I tested this project on Python 3.5, CUDA 9.0, cudnn 7.3.1 and other common packages listed in `requirements.txt`.

# Installation
### 1. Clone this repository

### 2. Install dependencies 
``` pip install -r requirements.txt```

# Sample test
### Run  'steering_on_video.py'

![test](./images/test1.gif)

# Train
### Dataset

I considered the following dataset lists.

(1) Image to segementation
https://bdd-data.berkeley.edu/
https://research.mapillary.com/

(2) Segementation to steer
https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip
https://drive.google.com/file/d/0B-KJCaaF7elleG1RbzVPZWV4Tlk/view

### Image preprocessing

I provide some tools for image preprocessing.

'image_resize.py' : Change image size to 180X60
'mask_selector.py' : If your datasets have more than one label, select the color of the label.
'segmentation_generator.py' : Create a segmentation of the image using the learned model. It is used to derive the steering value from the segmentation.

### Training

You can choose either train1 or train2 to learn. train2 is more advanced than train1. 
'load_data_from_segmentation.py' and 'load_data_from_steer.py' are the process of making each data into a pickle file.

### Other

'segmentation_generate_and_get_steer-optical flow.py' of train2 can derive the steer value by optical flow when there is no information of the steer value of the image.
