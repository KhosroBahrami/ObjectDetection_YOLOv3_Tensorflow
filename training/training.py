
# training of YOLOv3 model using a given dataset

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from configs.config_train import *
import tensorflow.contrib.slim.nets 


class Training(object):

        def __init__(self):
                a=1

        def training(self, sess, model, images, is_training, y_true):

                with tf.variable_scope('yolov3'):
                    pred_feature_map = model.forward(images, is_training=is_training)
                    loss             = model.compute_loss(pred_feature_map, y_true)
                    y_pred           = model.predict(pred_feature_map)

                tf.summary.scalar("loss/coord_loss",   loss[1])
                tf.summary.scalar("loss/sizes_loss",   loss[2])
                tf.summary.scalar("loss/confs_loss",   loss[3])
                tf.summary.scalar("loss/class_loss",   loss[4])

                global_step = tf.Variable(0, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
                write_op = tf.summary.merge_all()
                writer_train = tf.summary.FileWriter(FLAGS.train_summary_data_path)

                saver_to_restore = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(
                                                                         include=[FLAGS.train_darknet_model_path]))
                update_vars = tf.contrib.framework.get_variables_to_restore(include=[FLAGS.train_yolov3_model_path])
                learning_rate = tf.train.exponential_decay(FLAGS.train_learning_rate, global_step,
                                                           decay_steps=FLAGS.train_decay_steps,
                                                           decay_rate=FLAGS.train_decay_rate, staircase=True)
                optimizer = tf.train.AdamOptimizer(learning_rate)

                # set dependencies for BN ops
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.minimize(loss[0], var_list=update_vars, global_step=global_step)

                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                saver_to_restore.restore(sess, FLAGS.train_yolov3_checkpoints)
                saver = tf.train.Saver(max_to_keep=2)


                for epoch in range(FLAGS.train_epochs):
                    run_items = sess.run([train_op, write_op, y_pred, y_true] + loss, feed_dict={is_training:True})

                    if (epoch+1) % FLAGS.train_eval_internal == 0:
                        train_rec_value, train_prec_value = utils.evaluate(run_items[2], run_items[3])

                    writer_train.add_summary(run_items[1], global_step=epoch)
                    writer_train.flush() 
                    if (epoch+1) % 500 == 0: saver.save(sess, save_path=FLAGS.train_yolov3_checkpoints, global_step=epoch+1)

                    print("=> EPOCH %10d [TRAIN]:\tloss_xy:%7.4f \tloss_wh:%7.4f \tloss_conf:%7.4f \tloss_class:%7.4f"
                        %(epoch+1, run_items[5], run_items[6], run_items[7], run_items[8]))





