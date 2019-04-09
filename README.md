# YOLOv3 Object Detector based on Tensorflow


A TensorFlow implementation of YOLOv3 for object detection. 

It has been originally published in this research [paper](https://arxiv.org/abs/1804.02767). 
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
For training & testing, I used Pascal datasets. 
To prepare the datasets:
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
$ python3 convert_weight.py
```
The resulted tf records will be stored into tfrecords_test and tfrecords_train folders.

## Configuration
Before running the code, you need to touch the configuration based on your needs. We have 3 config files in /configs:
- config_train.py: this file includes training parameters.  
- config_test.py: this file includes testing parameters.  
- config_demo.py: this file includes demo parameters.  


## Demo of YOLOv3
Demo uses the pretrained model that has been stored in /checkpoints/yolov3... .

To run the demo, use the following command:
```python
# Run demo of YOLOv3 for one image
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
Test module uses the pretrained model that has been stored in /checkpoints/yolov3... . 

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




# How YOLOv1, YOLOv2, and YOLOv3 work?

YOLO (You Only Look Once), is a network for object detection targeted for real-time processing. 
The object detection is the task of determining the location of certain objects, as well as classifying those objects. It's first version was YOLOv1 and after some improvements, finally the last one proposed as YOLOv3.

The following table compares YOLOv1, YOLOv2 and YOLOv3.

| Object Detection Method | VOC2007 test mAP |  Speed (FPS) | Number of Prior Boxes | Input Resolution |
| :---: |   :---:     | :---: | :---: | :---: |
| YOLOv1 (19  layers) | 63.4% | 45 |  98 | 448*448 |
| YOLOv2 (30  layers) |  |  | 845  | 416*416 |
| YOLOv3 (106 layers) |  |  |  10,647 | 416*416 |




# YOLOv1

YOLO divides the input image into an S×S grid. Each grid cell predicts only one object. For example, the yellow grid cell in the folowing figure tries to predict the person object whose center (the blue dot) falls inside the yellow grid cell.

![Alt text](figs/fig1.png?raw=true "Each grid cell detects only one object.")

Each grid cell predicts a fixed number of boundary boxes. In this example, the yellow grid cell makes two boundary box predictions (blue boxes) to locate where the person is.
However, the one-object rule limits how close detected objects can be. For that, YOLO does have some limitations on how close objects can be.
![Alt text](figs/fig2.png?raw=true "Each grid cell make a fixed number of boundary box guesses for the object.")

For each grid cell,
- it predicts B boundary boxes and each box has one box confidence score,
- it detects one object only regardless of the number of boxes B,
- it predicts C conditional class probabilities (one per class for the likeliness of the object class).

To evaluate PASCAL VOC, YOLO uses 7×7 grids (S×S), 2 boundary boxes (B) and 20 classes (C).

Each boundary box contains 5 elements: (x, y, w, h) and a box confidence score. The confidence score reflects how likely the box contains an object (objectness) and how accurate is the boundary box. We normalize the bounding box width w and height h by the image width and height. x and y are offsets to the corresponding cell. Hence, x, y, w and h are all between 0 and 1. Each cell has 20 conditional class probabilities. The conditional class probability is the probability that the detected object belongs to a particular class (one probability per category for each cell). So, YOLO’s prediction has a shape of (S, S, B×5 + C) = (7, 7, 2×5 + 20) = (7, 7, 30).

![Alt text](figs/netv1.png?raw=true "YOLOv1 Network")[1]

The major concept of YOLO is to build a CNN network to predict a (7, 7, 30) tensor. It uses a CNN network to reduce the spatial dimension to 7×7 with 1024 output channels at each location. YOLO performs a linear regression using two fully connected layers to make 7×7×2 boundary box predictions (the middle picture below). To make a final prediction, we keep those with high box confidence scores (greater than 0.25) as our final predictions (the right picture).

![Alt text](figs/ev1.png?raw=true "example")[1]

YOLO has 24 convolutional layers followed by 2 fully connected layers (FC). Some convolution layers use 1 × 1 reduction layers alternatively to reduce the depth of the features maps. For the last convolution layer, it outputs a tensor with shape (7, 7, 1024). The tensor is then flattened. Using 2 fully connected layers as a form of linear regression, it outputs 7×7×30 parameters and then reshapes to (7, 7, 30), i.e. 2 boundary box predictions per location.

A faster but less accurate version of YOLO, called Fast YOLO, uses only 9 convolutional layers with shallower feature maps.


### Loss function

YOLO predicts multiple bounding boxes per grid cell. To compute the loss for the true positive, we only want one of them to be responsible for the object. For this purpose, we select the one with the highest IoU (intersection over union) with the ground truth. This strategy leads to specialization among the bounding box predictions. Each prediction gets better at predicting certain sizes and aspect ratios.

YOLO uses sum-squared error between the predictions and the ground truth to calculate loss. The loss function composes of:
- the classification loss.
- the localization loss (errors between the predicted boundary box and the ground truth).
- the confidence loss (the objectness of the box).

The final loss adds localization, confidence and classification losses together, as
![Alt text](figs/loss.png?raw=true "Loss function")



### Non Maxmimum Supression (NMS)

Due to multiple anchors, YOLO can make duplicate detections for the same object. To remove duplications with lower confidence, YOLO applies non-maximal suppression, as follow: 
- Sort the predictions by the confidence scores.
- Start from the top scores, ignore any current prediction if we find any previous predictions that have the same class and IoU > 0.5 with the current prediction.
- Repeat step 2 until all predictions are checked.



----------------------------

# YOLOv2
YOLOv2 is the second version of YOLO and has some contributions and improvements. 

### Batch normalization

YOLOv2 has batch normalization in convolution layers. This removes the need for dropouts and pushes mAP up 2%.

### High-resolution classifier

The YOLOv1 training composes of 2 phases. First, we train a classifier network like VGG16. Then we replace the fully connected layers with a convolution layer and retrain it end-to-end for the object detection. YOLOv1 trains the classifier with 224 × 224 pictures followed by 448 × 448 pictures for the object detection. YOLOv2 starts with 224 × 224 pictures for the classifier training but then retune the classifier again with 448 × 448 pictures using much fewer epochs. This makes the detector training easier and moves mAP up by 4%.

### Convolutional with Anchor Boxes
As indicated in the YOLO paper, the early training is susceptible to unstable gradients. Initially, YOLO makes arbitrary guesses on the boundary boxes. These guesses may work well for some objects but badly for others resulting in steep gradient changes. In early training, predictions are fighting with each other on what shapes to specialize on.

In the real-life domain, the boundary boxes are not arbitrary. Cars have very similar shapes and pedestrians have an approximate aspect ratio of 0.41. Since we only need one guess to be right, the initial training will be more stable if we start with diverse guesses that are common for real-life objects. For example, we can create 5 anchor boxes with the following shapes.

Instead of predicting 5 arbitrary boundary boxes, we predict offsets to each of the anchor boxes above. If we constrain the offset values, we can maintain the diversity of the predictions and have each prediction focuses on a specific shape. So the initial training will be more stable.

![Alt text](figs/5anchors.png?raw=true "5 anchor boxes")




### Backbone network & Feature maps
The following figure shows the changes that has been done to the network in YOLOv2: Remove the fully connected layers responsible for predicting the boundary box.


![Alt text](figs/networkv2.png?raw=true "network of v2")


YOLOv2 moves the class prediction from the cell level to the boundary box level. Now, each prediction includes 4 parameters for the boundary box, 1 box confidence score (objectness) and 20 class probabilities. i.e. 5 boundary boxes with 25 parameters: 125 parameters per grid cell. Same as YOLOv1, the objectness prediction still predicts the IOU of the ground truth and the proposed box.

To generate predictions with a shape of 7 × 7 × 125, YOLOv2 replaces the last convolution layer with three 3 × 3 convolutional layers each outputting 1024 output channels. Then it applies a final 1 × 1 convolutional layer to convert the 7 × 7 × 1024 output into 7 × 7 × 125. 

Laso, the input image size was changed from 448 × 448 to 416 × 416. This creates an odd number spatial dimension (7×7 v.s. 8×8 grid cell). The center of a picture is often occupied by a large object. With an odd number grid cell, it is more certain on where the object belongs.

Also, remove one pooling layer to make the spatial output of the network to 13×13 (instead of 7×7).

Anchor boxes decrease mAP slightly from 69.5 to 69.2 but the recall improves from 81% to 88%. i.e. even the accuracy is slightly decreased but it increases the chances of detecting all the ground truth objects.


### Dimension Clusters
In many problems, the boundary boxes have patterns. For example, in the autonomous driving, the 2 most common boundary boxes will be cars and pedestrians at different distances. To identify the top-K boundary boxes that have the best coverage for the training data, in YOLOv2, K-means clustering is used on the training data to locate the centroids of the top-K clusters.


### Direct location prediction
YOLOv2 makes predictions on the offsets to the anchors. YOLO predicts 5 parameters (tx, ty, tw, th, and to) and applies the sigma function to constraint its possible offset range. Here is the visualization. The blue box below is the predicted boundary box and the dotted rectangle is the anchor.

![Alt text](figs/prediction.png?raw=true "prediction")


### Fine-Grained Features
Convolution layers decrease the spatial dimension gradually. As the corresponding resolution decreases, it is harder to detect small objects. Other object detectors like SSD locate objects from different layers of feature maps, so each layer specializes at a different scale. YOLOv2 adopts a different approach called passthrough. It reshapes the 28 × 28 × 512 layer to 14 × 14 × 2048. Then it concatenates with the original 14 × 14 ×1024 output layer. Now we apply convolution filters on the new 14 × 14 × 3072 layer to make predictions.


### Multi-Scale Training
After removing the fully connected layers, YOLOv2 can take images of different sizes. If the width and height are doubled, we are just making 4x output grid cells and therefore 4x predictions. Since the YOLO network downsamples the input by 32, we just need to make sure the width and height is a multiple of 32. During training, YOLO takes images of size 320×320, 352×352, … and 608×608 (with a step of 32). For every 10 batches, YOLOv2 randomly selects another image size to train the model. This acts as data augmentation and forces the network to predict well for different input image dimension and scale. In additional, we can use lower resolution images for object detection at the cost of accuracy. This can be a good tradeoff for speed on low GPU power devices. At 288 × 288 YOLOv2 runs at more than 90 FPS with mAP almost as good as Fast R-CNN. At high-resolution YOLOv2 achieves 78.6 mAP on VOC 2007.




----------------------------


# YOLOv3


### Backbone network & Feature maps

YOLOv2 used a custom deep architecture darknet-19, an originally 19-layer network supplemented with 11 more layers for object detection. With a 30-layer architecture, YOLOv2 often struggled with small object detections. This was attributed to loss of fine-grained features as the layers downsampled the input. To remedy this, YOLOv2 used an identity mapping, concatenating feature maps from a previous layer to capture low level features.

However, YOLOv2’s architecture was still lacking some of the most important elements that are now staple in most of state-of-the art algorithms. No residual blocks, no skip connections and no upsampling. YOLOv3 incorporates all of these.

First, YOLOv3 uses a variant of Darknet, which originally has 53 layer network trained on Imagenet. For the task of detection, 53 more layers are stacked onto it, giving us a 106 layer fully convolutional underlying architecture for YOLOv3. This is the reason behind the slowness of YOLOv3 compared to YOLOv2. Here is how the architecture of YOLOv3 looks like.

![Alt text](figs/YOLOv3_network.png?raw=true "YOLOv3 Network")


The newer architecture boasts of residual skip connections, and upsampling. The most salient feature of YOLOv3 is that it makes detections at three different scales. YOLO is a fully convolutional network and its eventual output is generated by applying a 1 x 1 kernel on a feature map. In YOLOv3, the detection is done by applying 1 x 1 detection kernels on feature maps of three different sizes at three different places in the network.

The shape of the detection kernel is 1 x 1 x (B x (5 + C) ). Here B is the number of bounding boxes a cell on the feature map can predict, “5” is for the 4 bounding box attributes and one object confidence, and C is the number of classes. In YOLOv3 trained on COCO, B = 3 and C = 80, so the kernel size is 1 x 1 x 255. The feature map produced by this kernel has identical height and width of the previous feature map, and has detection attributes along the depth as described above.

![Alt text](figs/image_grid.png?raw=true "Image Grid")

In YOLOv3, the stride of the network, or a layer is defined as the ratio by which it downsamples the input. 

YOLOv3 makes prediction at three scales, which are precisely given by downsampling the dimensions of the input image by 32, 16 and 8 respectively.

The first detection is made by the 82nd layer. For the first 81 layers, the image is down sampled by the network, such that the 81st layer has a stride of 32. If we have an image of 416 x 416, the resultant feature map would be of size 13 x 13. One detection is made here using the 1 x 1 detection kernel, giving us a detection feature map of 13 x 13 x 255.

Then, the feature map from layer 79 is subjected to a few convolutional layers before being up sampled by 2x to dimensions of 26 x 26. This feature map is then depth concatenated with the feature map from layer 61. Then the combined feature maps is again subjected a few 1 x 1 convolutional layers to fuse the features from the earlier layer (61). Then, the second detection is made by the 94th layer, yielding a detection feature map of 26 x 26 x 255.

A similar procedure is followed again, where the feature map from layer 91 is subjected to few convolutional layers before being depth concatenated with a feature map from layer 36. Like before, a few 1 x 1 convolutional layers follow to fuse the information from the previous layer (36). We make the final of the 3 at 106th layer, yielding feature map of size 52 x 52 x 255.

Detections at different layers helps address the issue of detecting small objects, a frequent complaint with YOLOv2. The upsampled layers concatenated with the previous layers help preserve the fine grained features which help in detecting small objects.

The 13 x 13 layer is responsible for detecting large objects, whereas the 52 x 52 layer detects the smaller objects, with the 26 x 26 layer detecting medium objects. Here is a comparative analysis of different objects picked in the same object by different layers.


### Prior boxes (Anchor points) 
YOLOv3, in total uses 9 anchor boxes. Three for each scale. If you’re training YOLOv3 on your own dataset, you should go about using K-Means clustering to generate 9 anchors.

Then, arrange the anchors is descending order of a dimension. Assign the three biggest anchors for the first scale , the next three for the second scale, and the last three for the third.

For an input image of same size, YOLOv3 predicts more bounding boxes than YOLOv2. For instance, at it’s native resolution of 416 x 416, YOLOv2 predicted 13 x 13 x 5 = 845 boxes. At each grid cell, 5 boxes were detected using 5 anchors.

On the other hand YOLOv3 predicts boxes at 3 different scales. For the same image of 416 x 416, the number of predicted boxes are 10,647. This means that YOLOv3 predicts 10x the number of boxes predicted by YOLOv2. You could easily imagine why it’s slower than YOLOv2. At each scale, every grid can predict 3 boxes using 3 anchors. Since there are three scales, the number of anchor boxes used in total are 9, 3 for each scale.



### Loss function

YOLOv2’s loss function is defined as follow.

![Alt text](figs/loss.png?raw=true "Loss function")

It has 5 terms. The 3'th one penalizes the objectness score prediction for bounding boxes responsible for predicting objects (the scores for these should ideally be 1), the 4'th one for bounding boxes having no objects, (the scores should ideally be zero), and the last one penalises the class prediction for the bounding box which predicts the objects.
The last three terms in YOLOv2 are the squared errors, whereas in YOLOv3, they’ve been replaced by cross-entropy error terms. In other words, object confidence and class predictions in YOLOv3 are now predicted through logistic regression.

While we are training the detector, for each ground truth box, we assign a bounding box, whose anchor has the maximum overlap with the ground truth box.

YOLOv3 performs multilabel classification for objects detected in images.

In YOLOv2, authors used to softmax the class scores and take the class with maximum score to be the class of the object contained in the bounding box. This has been modified in YOLOv3.

Softmaxing classes rests on the assumption that classes are mutually exclusive, or in simple words, if an object belongs to one class, then it cannot belong to the other. This works fine in COCO dataset.

However, when we have classes like Person and Women in a dataset, then the above assumption fails. This is the reason why the authors of YOLO have refrained from softmaxing the classes. Instead, each class score is predicted using logistic regression and a threshold is used to predict multiple labels for an object. Classes with scores higher than this threshold are assigned to the box.

-----------------------------


### Class Prediction
Most classifiers assume output labels are mutually exclusive. It is true if the output are mutually exclusive object classes. Therefore, YOLOv2 applies a softmax function to convert scores into probabilities that sum up to one. YOLOv3 uses multi-label classification. For example, the output labels may be “pedestrian” and “child” which are not non-exclusive. (the sum of output can be greater than 1 now.) YOLOv3 replaces the softmax function with independent logistic classifiers to calculate the likeliness of the input belongs to a specific label. Instead of using mean square error in calculating the classification loss, YOLOv3 uses binary cross-entropy loss for each label. This also reduces the computation complexity by avoiding the softmax function.


### Bounding box prediction & cost function calculation
YOLOv3 predicts an objectness score for each bounding box using logistic regression. YOLOv3 changes the way in calculating the cost function. If the bounding box prior (anchor) overlaps a ground truth object more than others, the corresponding objectness score should be 1. For other priors with overlap greater than a predefined threshold (default 0.5), they incur no cost. Each ground truth object is associated with one boundary box prior only. If a bounding box prior is not assigned, it incurs no classification and localization lost, just confidence loss on objectness. We use tx and ty (instead of bx and by) to compute the loss.
![Alt text](figs/lossv3.png?raw=true "")


### Feature Pyramid Networks (FPN) like Feature Pyramid

YOLOv3 makes 3 predictions per location. Each prediction composes of a boundary box, a objectness and 80 class scores, i.e. N × N × [3 × (4 + 1 + 80) ] predictions.

YOLOv3 makes predictions at 3 different scales (similar to the FPN):

- In the last feature map layer.
- Then it goes back 2 layers back and upsamples it by 2. YOLOv3 then takes a feature map with higher resolution and merge it with the upsampled feature map using element-wise addition. YOLOv3 apply convolutional filters on the merged map to make the second set of predictions.
- Repeat 2 again so the resulted feature map layer has good high-level structure (semantic) information and good resolution spatial information on object locations.

To determine the priors, YOLOv3 applies k-means cluster. Then it pre-select 9 clusters. For COCO, the width and height of the anchors are (10×13),(16×30),(33×23),(30×61),(62×45),(59× 119),(116 × 90),(156 × 198),(373 × 326). These 9 priors are grouped into 3 different groups according to their scale. Each group is assigned to a specific feature map above in detecting objects.


### Feature extractor
In YOLOv3, the 53-layer Darknet-53 is used to replace the Darknet-19 as the feature extractor. Darknet-53 mainly compose of 3 × 3 and 1× 1 filters with skip connections like the residual network in ResNet. Darknet-53 has less BFLOP (billion floating point operations) than ResNet-152, but achieves the same classification accuracy at 2x faster.

![Alt text](figs/darknet53.png?raw=true "darknet53")









### References:
[1] https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088

[2] https://datascience.stackexchange.com/questions/25119/how-to-calculate-map-for-detection-task-for-the-pascal-voc-challenge

[3] https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b

[4] https://www.kdnuggets.com/2018/05/implement-yolo-v3-object-detector-pytorch-part-1.html




