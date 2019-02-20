# YOLOv3 Object Detector based on Tensorflow


A TensorFlow implementation of the YOLOv3 for object detection. 

It has been originally published in this research [paper](https://arxiv.org/abs/1512.02325). 
This repository contains a TensorFlow re-implementation of YOLOv3 which is inspired by the previous caffe 
and tensorflow implementations. However, this code has clear pipelines for train, test and demo; 
it is modular that can be extended or be use for new applications.



## Introduction
This implementation of You Look Only Once (YOLO) for object detection based on tensorflow is designed with 
the following goals:
- Pipeline: it has full pipeline of object detection for demo, test and train with seperate modules.
- Modularity: This code is modular and easy to expand for any specific application or new ideas.


## Prerequisite
The main requirements can be installed by:

```bash
pip install tensorflow    # For CPU
pip install tensorflow-gpu  # For GPU

# Install opencv for preprocessing training examples
pip install opencv-python
```

## Datasets
For training & testing, I used Pascal Pascal datasets. 
To prapare tha datasets:
1. Download VOC2007 and VOC2012 datasets. I assume the data is stored in /datasets/
```
$ cd datasets
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```

2. Convert the data to Tensorflow records:
```
$ tar -xvf VOCtrainval_11-May-2012.tar
$ tar -xvf VOCtrainval_06-Nov-2007.tar
$ tar -xvf VOCtest_06-Nov-2007.tar
$ python3 ssd_image_to_tf.py
```
The resulted tf records will be stored into tfrecords_test and tfrecords_train folders.

## Configuration
Before running the code, you need to touch the configuration based on your needs. We have 3 config files in /configs:
- config_train.py: this file includes training parameters.  
- config_test.py: this file includes testing parameters.  
- config_demo.py: this file includes demo parameters.  


## Demo of YOLOv3
Demo uses the pretrained model that has been stored in /checkpoints/ssd_... .

To run the demo, use the following command:
```python
# Run demo of SSD for one image
python3 yolo_v3_demo.py
```
The demo module has the following 6 steps:
1) Define a placeholder for the input image 
2) Preprocessing step
3) Create YOLOv3 model
4) Inference, calculate output of network
5) Postprocessing
6) Visualization & Evaluation


## Evaluating (Testing) YOLOv3 
Test module uses the pretrained model that has been stored in /checkpoints/ssd_... . 

To test the YOLOv3, use the following command:
```python
# Run test of YOLOv3
python3 yolo_v3_test.py
```
Evaluation module has the following 6 steps:
1) Data preparation
2) Preprocessing step
3) Create YOLOv3 model
4) Inference, calculate output of network
5) Postprocessing        
6) Evaluation




## Training YOLOv3
The input of training should be in /checkpoints/[model_name]
the output of training will be store in checkpoints/yolo_v3_[model_name]

To train the YOLOv3, use the following command:
```python
# Run training of YOLOv3
python3 yolo_v3_train.py
```

The Training module has the following 3 steps:
1) Data preparation
2) Preprocessing step
3) Training




# How SSD works?
