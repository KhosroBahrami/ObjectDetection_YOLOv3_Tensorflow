# Postprocessing


import colorsys
import numpy as np
import tensorflow as tf
from collections import Counter
from PIL import ImageFont, ImageDraw, Image

from configs.config_test import *


class Postprocessing(object):

    def __init__(self):      
        a=1




    def py_nms(self, boxes, scores, max_boxes=50, iou_thresh=0.5):
        """
        Pure Python NMS baseline.

        Arguments: boxes => shape of [-1, 4], the value of '-1' means that dont know the
                            exact number of boxes
                   scores => shape of [-1,]
                   max_boxes => representing the maximum of boxes to be selected by non_max_suppression
                   iou_thresh => representing iou_threshold for deciding to keep boxes
        """
        assert boxes.shape[1] == 4 and len(scores.shape) == 1

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]

        return keep[:max_boxes]




    def non_maximum_supression(self, boxes, scores, num_classes, max_boxes=50, score_thresh=0.3, iou_thresh=0.5):
        """
        /*----------------------------------- NMS on cpu ---------------------------------------*/
        Arguments:
            boxes ==> shape [1, 10647, 4]
            scores ==> shape [1, 10647, num_classes]
        """

        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1, num_classes)
        # Picked bounding boxes
        picked_boxes, picked_score, picked_label = [], [], []

        for i in range(num_classes):
            indices = np.where(scores[:,i] >= score_thresh)
            filter_boxes = boxes[indices]
            filter_scores = scores[:,i][indices]
            if len(filter_boxes) == 0: continue
            # do non_max_suppression on the cpu
            indices = self.py_nms(filter_boxes, filter_scores,
                             max_boxes=max_boxes, iou_thresh=iou_thresh)
            picked_boxes.append(filter_boxes[indices])
            picked_score.append(filter_scores[indices])
            picked_label.append(np.ones(len(indices), dtype='int32')*i)
        if len(picked_boxes) == 0: return None, None, None

        boxes = np.concatenate(picked_boxes, axis=0)
        score = np.concatenate(picked_score, axis=0)
        label = np.concatenate(picked_label, axis=0)

        return boxes, score, label

 
   
    def box_detection(self, sess, y_pred_tensor, y_true_tensor, images_tensor, NUM_CLASSES):

       for image_idx in range(3): #FLAGS.test_number_of_test_images):    
            y_pred, y_true, image  = sess.run([y_pred_tensor, y_true_tensor, images_tensor])
            
            pred_boxes = y_pred[0][0]
            pred_confs = y_pred[1][0]
            pred_probs = y_pred[2][0]
            image = Image.fromarray(np.uint8(image[0]*255))

            true_labels_list, true_boxes_list = [], []
            for i in range(3):
                true_probs_temp = y_true[i][..., 5: ]
                true_boxes_temp = y_true[i][..., 0:4]
                object_mask     = true_probs_temp.sum(axis=-1) > 0

                true_probs_temp = true_probs_temp[object_mask]
                true_boxes_temp = true_boxes_temp[object_mask]

                true_labels_list += np.argmax(true_probs_temp, axis=-1).tolist()
                true_boxes_list  += true_boxes_temp.tolist()

            pred_boxes, pred_scores, pred_labels = self.non_maximum_supression(pred_boxes,
                                                          pred_confs*pred_probs, NUM_CLASSES,
                                                          score_thresh=FLAGS.test_score_threshold, iou_thresh=FLAGS.test_iou_threshold)
            #image = utils.draw_boxes(image, pred_boxes, pred_scores, pred_labels, CLASSES, [FLAGS.test_image_height, FLAGS.test_image_width], show=True)

            true_boxes = np.array(true_boxes_list)
            box_centers, box_sizes = true_boxes[:,0:2], true_boxes[:,2:4]

            true_boxes[:,0:2] = box_centers - box_sizes / 2.
            true_boxes[:,2:4] = true_boxes[:,0:2] + box_sizes
            pred_labels_list = [] if pred_labels is None else pred_labels.tolist()

            all_detections   = []
            all_annotations  = []
            all_detections.append( [pred_boxes, pred_scores, pred_labels_list])
            all_annotations.append([true_boxes, true_labels_list])
            print('image_idx:', image_idx)

       return all_detections, all_annotations













