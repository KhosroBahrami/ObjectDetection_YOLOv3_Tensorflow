
# Show checkpoints
from tensorflow.python import pywrap_tensorflow
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets 


print('\n yolov3:')
reader = pywrap_tensorflow.NewCheckpointReader('checkpoint/yolov3.ckpt')
var_to_shape_map = reader.get_variable_to_shape_map()
#saver = tf.train.Saver()
print('\n')
for v in sorted(var_to_shape_map):
   print(v)


print('\n\n yolov3 2500 :')
reader = pywrap_tensorflow.NewCheckpointReader('checkpoint/yolov3.ckpt-2500')
var_to_shape_map = reader.get_variable_to_shape_map()
#saver = tf.train.Saver()
print('\n')
for v in sorted(var_to_shape_map):
   print(v)



   
