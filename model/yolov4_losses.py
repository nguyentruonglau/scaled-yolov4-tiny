import tensorflow as tf
from generator.config_data import yolo_anchors
import numpy as np
import math


def yolov4_loss(args, grid_index):
    """Use for training phase, total loss = object loss + regression loss + classification loss
    """
    if args.box_regression_loss == 'diou':
        box_regression_loss = BoxRegressionLoss.ciou(args, grid_index, 'diou')
    elif args.box_regression_loss == 'giou':
        box_regression_loss = BoxRegressionLoss.ciou(args, grid_index, 'giou')
    else: box_regression_loss = BoxRegressionLoss.ciou(args, grid_index, 'ciou')

    if args.classification_loss == 'sigmoid_focal':
        classification_loss = ClassificationLoss.focal(args, grid_index)
    elif args.classification_loss == 'bce':
        classification_loss = ClassificationLoss.bce(args, grid_index)
    else: raise ValueError('[ERROR]: losses function not found')

    object_loss = ObjectLoss.bce(args, grid_index)

    def loss(y_true, y_pred):
        model_obj_loss_layers_weights=[4.0, 1.0]
        obj_loss_layers_weights = model_obj_loss_layers_weights[grid_index]

        detect_layer_num = len(model_obj_loss_layers_weights)
        model_loss_scale = 3 / detect_layer_num
        box_reg_losss_weight = 0.05 * model_loss_scale
        obj_losss_weight = 1.0 * model_loss_scale * (1.4 if detect_layer_num >= 4 else 1.)
        cls_losss_weight = 0.5 * model_loss_scale

        box_reg_loss, iou_score = box_regression_loss(y_true, y_pred)
        obj_loss = object_loss(y_true, y_pred, iou_score)*obj_loss_layers_weights
        cls_loss = classification_loss(y_true, y_pred)

        if int(args.num_classes) == 1:
            return box_reg_losss_weight*box_reg_loss + obj_losss_weight*obj_loss
        return  box_reg_losss_weight*box_reg_loss + obj_losss_weight*obj_loss + cls_losss_weight*cls_loss

    return loss


class ClassificationLoss():
    """This loss function will answer the question, what class does the object in the box belong to?
    """
    @staticmethod
    #Used when there are many objects to detect
    def focal(args, grid_index):
        def loss(y_true, y_pred):
            object_mask = y_true[..., 4]
            y_true = y_true[..., 5:]
            y_pred = y_pred[..., 5:]
            y_pred = tf.math.sigmoid(y_pred)
            alpha_weight = tf.where(y_true != 0, 0.25, 0.75)
            focal_weight = tf.where(y_true != 0, 1. - y_pred, y_pred)
            focal_weight = alpha_weight * focal_weight ** 2.0
            bce_loss = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
            focal_loss = focal_weight*bce_loss
            focal_loss = tf.reduce_sum(focal_loss, axis=[-1])
            focal_loss = tf.reduce_sum(object_mask*focal_loss,axis=[1, 2, 3])
            normalizer = tf.maximum(tf.reduce_sum(object_mask, axis=[1, 2, 3]), 1.0)
            return focal_loss/normalizer
        return loss

    @staticmethod
    #Used when there is one object to detect
    def bce(args, grid_index):
        def loss(y_true, y_pred):
            object_mask = y_true[..., 4]
            pos_num = tf.reduce_sum(object_mask, axis=[1,2,3])
            pos_num = tf.maximum(pos_num, 1)
            classification_loss = object_mask * tf.keras.losses.binary_crossentropy(y_true[..., 5:],y_pred[..., 5:],from_logits=True,label_smoothing=args.label_smooth)
            return tf.reduce_sum(classification_loss, axis=[1,2,3])/pos_num
        return loss



class ObjectLoss():
    """ Object Loss
    This loss function will answer the question, is there an object in the box?
    """
    @staticmethod
    def bce(args, grid_index):
        def loss(y_true, y_pred, iou_score):

            object_loss_mask = y_true[..., 4]
            iou_score = tf.maximum(iou_score, 0)
            iou_score = tf.stop_gradient(iou_score)
            object_loss_ori = tf.keras.losses.binary_crossentropy(tf.expand_dims(iou_score,axis=-1),
                                                                  y_pred[..., 4:5], from_logits=True)
            return tf.reduce_mean(object_loss_ori, axis=[1, 2, 3])
        return loss


class BoxRegressionLoss():
    """ Box Regression Loss, we have: iou loss, giou loss, diou loss, ciou loss
    This loss function will answer the question, where is the position of the object in the image?
    """
    @staticmethod
    def ciou(args, grid_index, type):
        def loss(y_true, y_pred):

            if args.model_type=='S':
                stride_base = 16
            else:
                stride_base = 8

            obj_mask = y_true[..., 4]
            pos_num = tf.reduce_sum(obj_mask,axis=[1,2,3])
            pos_num = tf.maximum(pos_num, 1)

            grid_height = tf.shape(y_true)[1]
            grid_width = tf.shape(y_true)[2]
            grid_xy = tf.stack(tf.meshgrid(tf.range(grid_width), tf.range(grid_height)), axis=-1)
            grid_xy = tf.reshape(grid_xy, [tf.shape(grid_xy)[0], tf.shape(grid_xy)[1], 1, tf.shape(grid_xy)[2]])
            grid_xy = tf.cast(grid_xy, tf.dtypes.float32)

            y_true_cxcy = grid_xy+y_true[...,0:2]
            y_true_x1y1x2y2 = tf.concat([y_true_cxcy - y_true[..., 2:4] / 2,y_true_cxcy + y_true[..., 2:4] / 2],axis=-1)

            scales_x_y = tf.cast(args.scales_x_y[grid_index], tf.dtypes.float32)
            scaled_pred_xy = tf.sigmoid(y_pred[..., 0:2]) * scales_x_y - 0.5 * (scales_x_y - 1)
           
            pred_xy = (grid_xy + scaled_pred_xy)
            normalized_anchors = tf.cast(yolo_anchors[args.model_type][grid_index], tf.dtypes.float32) / tf.cast(
                stride_base * (2 ** grid_index), tf.dtypes.float32)
            pred_wh = (tf.sigmoid(y_pred[..., 2:4]) * 2) ** 2 * normalized_anchors
            pred_x1y1x2y2 = tf.concat([pred_xy-pred_wh/2, pred_xy+pred_wh/2], axis=-1)

            #iou loss
            inter_w = tf.maximum(tf.minimum(pred_x1y1x2y2[..., 2], y_true_x1y1x2y2[..., 2]) - tf.maximum(pred_x1y1x2y2[..., 0], y_true_x1y1x2y2[..., 0]), 0)
            inter_h = tf.maximum(tf.minimum(pred_x1y1x2y2[..., 3], y_true_x1y1x2y2[..., 3]) - tf.maximum(pred_x1y1x2y2[..., 1], y_true_x1y1x2y2[..., 1]), 0)
            inter_area = inter_w * inter_h
            pred_wh = pred_x1y1x2y2[..., 2:4] - pred_x1y1x2y2[..., 0:2]
            y_true_wh = y_true_x1y1x2y2[..., 2:4] - y_true_x1y1x2y2[..., 0:2]
            boxes1_area = pred_wh[...,0] * pred_wh[...,1]
            boxes2_area = y_true_wh[..., 0] * y_true_wh[..., 1]
            union_area = boxes1_area + boxes2_area - inter_area + 1e-07

            iou_scores = inter_area / union_area

            box_center_dist = (pred_x1y1x2y2[..., 0:2] + pred_x1y1x2y2[..., 2:4])/2-(y_true_x1y1x2y2[..., 0:2] + y_true_x1y1x2y2[..., 2:4]) / 2
            box_center_dist = tf.reduce_sum(tf.math.square(box_center_dist), axis=-1)
            bounding_rect_wh = tf.maximum(pred_x1y1x2y2[..., 2:4], y_true_x1y1x2y2[..., 2:4])-tf.minimum(pred_x1y1x2y2[..., 0:2], y_true_x1y1x2y2[..., 0:2])
            diagonal_line_length = tf.reduce_sum(tf.math.square(bounding_rect_wh),axis=-1)
            diou_loss = iou_scores - box_center_dist/(diagonal_line_length+1e-07)

            #giou loss
            if type == 'giou':
                bounding_rect_area = bounding_rect_wh[..., 0] * bounding_rect_wh[...,1] + 1e-07
                giou_loss = iou_scores - (bounding_rect_area-union_area)/bounding_rect_area
                return tf.reduce_sum(obj_mask*(1-giou_loss),[1,2,3])/pos_num, giou_loss
            #diou loss
            if type == 'diou':
                return tf.reduce_sum(obj_mask*(1-diou_loss),[1,2,3])/pos_num, diou_loss

            #ciou loss
            y_true_wh = y_true_x1y1x2y2[..., 2:4] - y_true_x1y1x2y2[..., 0:2]
            v = (4. / math.pi ** 2) * tf.square(
                tf.math.atan2(pred_wh[..., 1], pred_wh[..., 0] + 1e-07) - tf.math.atan2(y_true_wh[..., 1],
                                                                                        y_true_wh[..., 0] + 1e-07))
            alpha = v/(1.-iou_scores+v+1e-07)

            alpha = tf.stop_gradient(alpha)
            ciou_loss = diou_loss - alpha*v
            return tf.reduce_sum(obj_mask*(1 - ciou_loss), [1,2,3])/pos_num,ciou_loss
        return loss