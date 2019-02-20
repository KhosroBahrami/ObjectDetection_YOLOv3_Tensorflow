
# Config of test
import tensorflow as tf
slim = tf.contrib.slim

tf.app.flags.DEFINE_integer('test_number_of_test_images', 20, 'number of test images')


tf.app.flags.DEFINE_integer('test_image_height', 416, 'height of input image')
tf.app.flags.DEFINE_integer('test_image_width', 416, 'width of input image')
tf.app.flags.DEFINE_integer('test_batch_size', 1, 'batch size')


# test on new dataset (x should be replaced with the name of new dataset)
tf.app.flags.DEFINE_string('test_image_names', './data/x.names', 'image names')
tf.app.flags.DEFINE_string('test_anchors', './data/x_anchors.txt', 'anchors')
tf.app.flags.DEFINE_string('test_tfrecords_path', './x_dataset/x_test.tfrecords', 'tfrecords path checkpoints')


tf.app.flags.DEFINE_string('test_yolov3_checkpoints', './checkpoint/yolov3.ckpt-2500', 'yolov3 checkpoints')

tf.app.flags.DEFINE_float('test_score_threshold', 0.3, 'score threshold')
tf.app.flags.DEFINE_float('test_iou_threshold', 0.5, 'iou threshold')



FLAGS = tf.app.flags.FLAGS


