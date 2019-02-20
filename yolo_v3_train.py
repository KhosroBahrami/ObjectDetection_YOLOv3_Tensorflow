
# This module trains the YOLOv3 object detection using a given dataset

import tensorflow as tf
from datasets.dataset import *
from networks import yolov3
from training.training import *
from configs.config_train import *




# main module for training of YOLOv3
def main():

    with tf.Session() as sess:


        # 1) Data preparation
        print('\n1) Data preparation...')
        oDataset = Dataset(FLAGS.train_train_tfrecord, FLAGS.train_batch_size, shuffle=None, repeat=False)
        CLASSES, NUM_CLASSES, ANCHORS = oDataset.read_class_anchors_attributes('train')
        parser = Parser(FLAGS.train_image_height, FLAGS.train_image_width, ANCHORS, NUM_CLASSES)

        trainset = Dataset(FLAGS.train_train_tfrecord, FLAGS.train_batch_size, shuffle=FLAGS.train_shuffle_size)
        trainset._buildup(parser)
        testset  = Dataset(FLAGS.train_test_tfrecord, FLAGS.train_batch_size, shuffle=None)
        testset._buildup(parser)
        is_training = tf.placeholder(tf.bool)
        images, *y_true = tf.cond(is_training, lambda: trainset.get_next(), lambda: testset.get_next())
        


        # 2) Create YOLOv3 model
        print('\n2) YOLOv3 model...')
        model = yolov3.yolov3(NUM_CLASSES, ANCHORS)


        # 3) Training
        print('\n3) Training...')
        oTraining = Training() 
        oTraining.training(sess, model, images, is_training, y_true)





if __name__ == '__main__':
    main()




