
# Config of training
import tensorflow as tf
slim = tf.contrib.slim


tf.app.flags.DEFINE_integer('train_image_height', 416, 'height of input image')
tf.app.flags.DEFINE_integer('train_image_width', 416, 'width of input image')

tf.app.flags.DEFINE_integer('train_batch_size', 8, 'batch size')
tf.app.flags.DEFINE_integer('train_epochs', 2500, 'number of epochs')
tf.app.flags.DEFINE_float('train_learning_rate', 0.001, 'learning rate')
tf.app.flags.DEFINE_integer('train_decay_steps', 100, 'decay steps')
tf.app.flags.DEFINE_float('train_decay_rate', 0.9, 'decay rate')

tf.app.flags.DEFINE_integer('train_shuffle_size', 200, 'shuffle size')
tf.app.flags.DEFINE_integer('train_eval_internal', 100, 'eval')

# train on new dataset (x should be replaced with the name of new dataset)
tf.app.flags.DEFINE_string('train_train_tfrecord', './x_dataset/x_train.tfrecords', 'train dataset')
tf.app.flags.DEFINE_string('train_test_tfrecord', './x_dataset/x_test.tfrecords', 'test dataset')
tf.app.flags.DEFINE_string('train_image_names', './data/x.names', 'image names')
tf.app.flags.DEFINE_string('train_anchors', './data/x_anchors.txt', 'anchors')


tf.app.flags.DEFINE_string('train_darknet_model_path', 'yolov3/darknet-53', 'darknet model path')
tf.app.flags.DEFINE_string('train_yolov3_model_path', 'yolov3/yolo-v3', 'yolov3 model path')
tf.app.flags.DEFINE_string('train_yolov3_checkpoints', './checkpoint/yolov3.ckpt', 'yolov3 checkpoints')

tf.app.flags.DEFINE_string('train_summary_data_path', './data/train', 'path of summary for data')



FLAGS = tf.app.flags.FLAGS



