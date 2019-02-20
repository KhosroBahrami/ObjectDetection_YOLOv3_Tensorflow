
# This module is used to load datasets for YOLO
import os
import tensorflow as tf
import sys
import random
import numpy as np

from configs.config_test import *
from datasets.parser import *
from configs.config_train import *


slim = tf.contrib.slim



class Dataset(object):


    def __init__(self, tfrecords_path, batch_size, shuffle=None, repeat=True):
        self.filenames = tf.gfile.Glob(tfrecords_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.repeat  = repeat

    def _buildup(self, parser):
        try:
            self._TFRecordDataset = tf.data.TFRecordDataset(self.filenames)
        except:
            raise NotImplementedError("No tfrecords found!")

        self._TFRecordDataset = self._TFRecordDataset.map(map_func = parser.parser_example,
                                                        num_parallel_calls = 10)
        self._TFRecordDataset = self._TFRecordDataset.repeat() if self.repeat else self._TFRecordDataset

        if self.shuffle is not None:
            self._TFRecordDataset = self._TFRecordDataset.shuffle(self.shuffle)

        self._TFRecordDataset = self._TFRecordDataset.batch(self.batch_size).prefetch(self.batch_size)
        self._iterator = self._TFRecordDataset.make_one_shot_iterator()



    def get_next(self):
        return self._iterator.get_next()





    # This function reads dataset from tfrecords
    # Inputs:
    #     datase_name: pascalvoc_2007
    #     train_or_test: test
    #     dataset_path: './tfrecords_test/'
    # Outputs:
    #     loaded dataset
    def read_class_anchors_attributes(self, mode):
        if mode=='train':
            CLASSES = self.read_coco_names(FLAGS.train_image_names)
            NUM_CLASSES = len(CLASSES)
            ANCHORS = self.get_anchors(FLAGS.train_anchors, FLAGS.train_image_height, FLAGS.train_image_width)
        elif mode=='test':
            CLASSES = self.read_coco_names(FLAGS.test_image_names)
            NUM_CLASSES = len(CLASSES)
            ANCHORS = self.get_anchors(FLAGS.test_anchors, FLAGS.test_image_height, FLAGS.test_image_width)
        
        return CLASSES, NUM_CLASSES, ANCHORS

 

    # This function reads dataset from tfrecords
    # Inputs:
    #     datase_name: pascalvoc_2007
    #     train_or_test: test
    #     dataset_path: './tfrecords_test/'
    # Outputs:
    #     loaded dataset
    def read_test_dataset(self, ANCHORS, NUM_CLASSES, batch_size=1, shuffle=None, repeat=False):
        #parser  = Parser(FLAGS.test_image_height, FLAGS.test_image_width, ANCHORS, NUM_CLASSES)
        #testset = dataset(parser, FLAGS.test_tfrecords_path , batch_size=1, shuffle=None, repeat=False)
        images_tensor, *y_true_tensor  = self.get_next()
        return images_tensor, (*y_true_tensor)


 
    def get_anchors(self, anchors_path, image_h, image_w):
        '''loads the anchors from a file'''
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = np.array(anchors.split(), dtype=np.float32)
        anchors = anchors.reshape(-1,2)
        anchors[:, 1] = anchors[:, 1] * image_h
        anchors[:, 0] = anchors[:, 0] * image_w
        return anchors.astype(np.int32)


    def read_coco_names(self, class_file_name):
        names = {}
        with open(class_file_name, 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names

        


