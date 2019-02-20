
# Evaluation of YOLOv3 object detection 
import math
import sys
import six
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python import debug as tf_debug
from configs.config_test import *

from tqdm import tqdm



class Evaluation(object):

    def __init__(self):
        a=1
        

    def evaluate(self, CLASSES, NUM_CLASSES, all_detections, all_annotations):
        
        for idx in range(NUM_CLASSES):
            print('idx: ',idx)
            true_positives  = []
            scores = []
            num_annotations = 0

            for i in tqdm(range(len(all_annotations)), desc="Computing AP for class %12s" %(CLASSES[idx])):
                print('i',i)
                pred_boxes, pred_scores, pred_labels_list = all_detections[i]
                true_boxes, true_labels_list = all_annotations[i]
                detected = []
                num_annotations += true_labels_list.count(idx)

                for k in range(len(pred_labels_list)):
                    if pred_labels_list[k] != idx: continue

                    scores.append(pred_scores[k])
                    ious = self.bbox_iou(pred_boxes[k:k+1], true_boxes)
                    m = np.argmax(ious)
                    if ious[m] > FLAGS.test_iou_threshold and pred_labels_list[k] == true_labels_list[m] and m not in detected:
                        detected.append(m)
                        true_positives.append(1)
                    else:
                        true_positives.append(0)

            num_predictions = len(true_positives)
            true_positives  = np.array(true_positives)
            false_positives = np.ones_like(true_positives) - true_positives
            # sorted by score
            indices = np.argsort(-np.array(scores))
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]
            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)
            # compute recall and precision
            recall    = true_positives / np.maximum(num_annotations, np.finfo(np.float64).eps)
            precision = true_positives / np.maximum(num_predictions, np.finfo(np.float64).eps)
            # compute average precision
            average_precision = self.compute_ap(recall, precision)
            all_aver_precs   = {CLASSES[i]:0. for i in range(NUM_CLASSES)}
            all_aver_precs[CLASSES[idx]] = average_precision

        for idx in range(NUM_CLASSES):
            cls_name = CLASSES[idx]
            print("=> Class %10s - AP: %.4f" %(cls_name, all_aver_precs[cls_name]))

        print("=> mAP: %.4f" %(sum(all_aver_precs.values()) / NUM_CLASSES))         
  

          

    def bbox_iou(self, A, B):

        intersect_mins = np.maximum(A[:, 0:2], B[:, 0:2])
        intersect_maxs = np.minimum(A[:, 2:4], B[:, 2:4])
        intersect_wh   = np.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        A_area = np.prod(A[:, 2:4] - A[:, 0:2], axis=1)
        B_area = np.prod(B[:, 2:4] - B[:, 0:2], axis=1)

        iou = intersect_area / (A_area + B_area - intersect_area)

        return iou



    def compute_ap(self, recall, precision):
        """ Compute the average precision, given the recall and precision curves.
        Code originally from https://github.com/rbgirshick/py-faster-rcnn.
        # Arguments
            recall:    The recall curve (list).
            precision: The precision curve (list).
        # Returns
            The average precision as computed in py-faster-rcnn.
        """
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap




    



