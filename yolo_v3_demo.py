
# Demo of object detection using YOLOv3 method

import time
import numpy as np
import tensorflow as tf
from PIL import Image
from datasets.dataset import *
from postprocessing.postprocessing import * 
from visualization.visualization import * 
from configs.config_demo import *
from configs.config_test import *



# main module for demo
def main():

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:

        # 1) Define a placeholder for the input image 
        print('\n1) Input image...')
        oDataset = Dataset(FLAGS.train_test_tfrecord, FLAGS.test_batch_size,shuffle=None, repeat=False)
        classes = oDataset.read_coco_names('./data/coco.names')
        img = Image.open(FLAGS.demo_path_of_demo_images)


        # 2) Preprocessing step
        print('\n2) Preprocessing...')
        img_resized = np.array(img.resize(size=(FLAGS.demo_image_height, FLAGS.demo_image_width)),
                                                                         dtype=np.float32)
        img_resized = img_resized / 255.


        # 3) Create YOLO model
        print('\n3) YOLOv3 model...')
        return_elements = ["Placeholder:0", "concat_9:0", "mul_6:0"]
        with tf.gfile.FastGFile(FLAGS.demo_frozen_graph_model_name, 'rb') as f:
            frozen_graph_def = tf.GraphDef()
            frozen_graph_def.ParseFromString(f.read())
        with graph.as_default():
            return_elements = tf.import_graph_def(frozen_graph_def, return_elements=return_elements)
            input_tensor, output_tensors = return_elements[0], return_elements[1:]



        # 4) Inference, calculate output of network
        print('\n4) YOLOv3 Inference...')
        boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})


        # 5) Postprocessing
        print('\n5) Postprocessing...')
        oPostprocess = Postprocessing()
        num_classes = len(classes)
        boxes, scores, labels = oPostprocess.non_maximum_supression(boxes, scores, num_classes,
                                                     score_thresh=FLAGS.demo_score_threshold,
                                                     iou_thresh=FLAGS.demo_iou_threshold)


        # 6) Visualization & Evaluation
        print('\n6) Visualization...')
        oVisualization = Visualization()
        image = oVisualization.draw_boxes(img, boxes, scores, labels, classes, [FLAGS.demo_image_height,
                                                                       FLAGS.demo_image_width], show=True)




if __name__ == '__main__':
    main()




    
