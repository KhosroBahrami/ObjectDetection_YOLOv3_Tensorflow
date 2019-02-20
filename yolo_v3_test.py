
# This module evaluates the YOLOv3 object detection
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
from datasets.dataset import *
from networks import yolov3
from postprocessing.postprocessing import * 
from evaluation.evaluation import *
from configs.config_test import *




# main module for evaluation of YOLOv3
def main():

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
       
    with tf.Session() as sess:
            
        # 1) Data preparation
        print('\n1) Data preparation...')
        oDataset = Dataset(FLAGS.train_test_tfrecord, FLAGS.test_batch_size,shuffle=None, repeat=False)

        CLASSES, NUM_CLASSES, ANCHORS = oDataset.read_class_anchors_attributes('test')
        
        parser  = Parser(FLAGS.test_image_height, FLAGS.test_image_width, ANCHORS, NUM_CLASSES)
        oDataset._buildup(parser)
        images_tensor, *y_true_tensor = oDataset.read_test_dataset(ANCHORS, NUM_CLASSES, batch_size=1,
                                                                   shuffle=None, repeat=False)


        # 2) Create YOLOv3 model
        print('\n2) YOLOv3 model...')
        model = yolov3.yolov3(NUM_CLASSES, ANCHORS)


        # 3) Inference, calculate output of network
        print('\n3) Inference...')
        with tf.variable_scope('yolov3'):
            pred_feature_map    = model.forward(images_tensor, is_training=False)
            y_pred_tensor       = model.predict(pred_feature_map)
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.test_yolov3_checkpoints)


        # 4) Postprocessing
        print('\n4) Postprocessing...')
        oPostprocess = Postprocessing()
        all_detections, all_annotations = oPostprocess.box_detection(sess, y_pred_tensor,
                                                                     y_true_tensor,
                                                                     images_tensor, NUM_CLASSES)


        # 5) Evaluation
        print('\n5) Evaluation...')
        oEvaluation = Evaluation()
        oEvaluation.evaluate(CLASSES, NUM_CLASSES, all_detections, all_annotations)






if __name__ == '__main__':
    main()



    


