
# Config for demo of YOLOv3 object detection
import tensorflow as tf
slim = tf.contrib.slim

tf.app.flags.DEFINE_integer('demo_image_height', 416, 'height of input image')
tf.app.flags.DEFINE_integer('demo_image_width', 416, 'width of input image')

tf.app.flags.DEFINE_string('demo_path_of_demo_images', './demo_images/road.jpg', 'path of demo images')
tf.app.flags.DEFINE_string('demo_frozen_graph_model_name', './checkpoint/yolov3_nms.pb', 'path of model')

tf.app.flags.DEFINE_float('demo_score_threshold', 0.5, 'score threshold')
tf.app.flags.DEFINE_float('demo_iou_threshold', 0.5, 'iou threshold')



FLAGS = tf.app.flags.FLAGS









































