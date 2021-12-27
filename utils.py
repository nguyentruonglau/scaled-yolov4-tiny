from model.scaled_yolov4_tiny import scaled_yolov4_tiny
from model.numpy_nms import NonMaxSuppression

import albumentations as aug
import tensorflow as tf

import os, cv2
import numpy as np


def tta_nms(boxes,scores,classes,valid_detections,args):
    """Test Time Augmentation - Non Maximum Suppression
    """
    all_boxes = []; all_scores = []; all_classes = []
    batch_index = 0

    valid_boxes = boxes[batch_index][0:valid_detections[batch_index]]
    valid_boxes[:, (0, 2)] = (1.-valid_boxes[:,(2,0)])

    all_boxes.append(valid_boxes)
    all_scores.append(scores[batch_index][0:valid_detections[batch_index]])
    all_classes.append(classes[batch_index][0:valid_detections[batch_index]])

    for batch_index in range(1,boxes.shape[0]):
        all_boxes.append(boxes[batch_index][0:valid_detections[batch_index]])
        all_scores.append(scores[batch_index][0:valid_detections[batch_index]])
        all_classes.append(classes[batch_index][0:valid_detections[batch_index]])

    all_boxes = np.concatenate(all_boxes,axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_classes = np.concatenate(all_classes, axis=0)

    all_boxes = np.array(all_boxes) 
    all_scores = np.array(all_scores)
    all_classes = np.array(all_classes)

    boxes, scores, classes, valid_detections = NonMaxSuppression.diou_nms_np_tta(
        np.expand_dims(all_boxes,0),
        np.expand_dims(all_scores,0),
        np.expand_dims(all_classes,0),
        args
    )
    boxes, scores = np.squeeze(boxes), np.squeeze(scores)
    classes, valid_detections = np.squeeze(classes), np.squeeze(valid_detections)
    
    return boxes[:valid_detections], scores[:valid_detections], classes[:valid_detections]


def get_tta_tranform():
    """Get Test Time Augmentation tranformation
       Given an input image, it will be transformed before entering the prediction model.
    """
    out_list=[]
    # flip horizontally and change color
    tta_transform = aug.Compose([
        aug.HorizontalFlip(p=1.),
        aug.RandomBrightnessContrast(p=0.2),
    ])
    # flip vertically and change color
    out_list.append(tta_transform)
    tta_transform = aug.Compose([
        aug.RandomBrightnessContrast(p=0.2),
    ])
    out_list.append(tta_transform)
    return out_list


def load_pretrained_model(args):
    """Load pretrained model on COCO DATASET
    """
    if args.use_pretrain:
        if not os.path.exists(args.pretrained_weights+'.index'):
            raise ValueError('[ERROR]: Pretrained weights not found!')

    model = scaled_yolov4_tiny(args, training=True)
    if not args.use_pretrain: return model

    try:
        model.load_weights(args.pretrained_weights).expect_partial()
        print("[INFOR]: Load {} checkpoints successfully!".format(args.model_type))
    except:
        cur_num_classes = int(args.num_classes)
        args.num_classes = 80
        pretrain_model = scaled_yolov4_tiny(args, training=True)
        pretrain_model.load_weights(args.pretrained_weights).expect_partial()
        for layer in model.layers:
            if not layer.get_weights():
                continue
            if 'yolov4_head' in layer.name:
                continue
            layer.set_weights(pretrain_model.get_layer(layer.name).get_weights())
        args.num_classes = cur_num_classes
        print("[INFOR]: Load {} weight successfully!".format(args.model_type))
    return model


def check_input_shape(args):
    """Check input shape of model
    """
    if args.model_shape % 32 != 0: raise ValueError('[ERROR]: Model shape not matched, \
        model-shape//32 must == 0 and >=320')
    return 0


def test_time_aug(args, img_ori, model, input_name, label_boxes, label_scores):
    """Apply test time augmentation data
    """
    # flip horizontally|vertical and change color
    img_copy = img_ori.copy()
    batch_img = []
    if args.test_time_aug:
        tta_transforms= get_tta_tranform()
        batch_img.append(tta_transforms[0](image=img_copy)['image'])
        batch_img.append(tta_transforms[1](image=img_copy)['image'])
    batch_img.append(img_copy)
    batch_img = np.array(batch_img, dtype=np.float32)
    # detect & apply nms
    boxes,scores,classes,valid_detections = detect_batch_img(
        args, batch_img, model,
        input_name, 
        label_boxes, label_scores
    )
    boxes, scores, classes = tta_nms(boxes, scores, classes, valid_detections, args)
    return boxes, scores, classes


def detect_batch_img(args, img, model, input_name, label_boxes, label_scores):
    """Detect batch image, using for test time augmentation
    """
    img = img / 255
    pre_nms_decoded_boxes, pre_nms__scores = model.run(
                            [label_boxes, label_scores],
                            {input_name: img}
                        )
    boxes, scores, classes, valid_detections = NonMaxSuppression.diou_nms_np(
                            pre_nms_decoded_boxes,
                            pre_nms__scores, args
                        )
    return boxes, scores, classes, valid_detections