from utility.coco_eval import CocoEvalidation
from model.scaled_yolov4_tiny import postprocess
from model.numpy_nms import NonMaxSuppression

from tqdm import tqdm
import os
import numpy as np


class EagerCocoMap():
    def __init__(self, pred_generator,model,args):
        self.args = args
        self.pred_generator = pred_generator
        self.model = model

        groundtruth_boxes = []; groundtruth_classes = []; groundtruth_valids = []
        with open(os.path.join(args.class_names)) as f:
            class_names = [name.strip() for name in f.readlines()]

        pred_generator_tqdm = tqdm(self.pred_generator, total=len(self.pred_generator))
        for batch_img, batch_boxes, batch_valids in pred_generator_tqdm:
            groundtruth_boxes.append(batch_boxes[..., 0:4])
            groundtruth_classes.append(batch_boxes[..., 4])
            groundtruth_valids.append(batch_valids)
            
        groundtruth_boxes = np.concatenate(groundtruth_boxes, axis=0)
        groundtruth_classes = np.concatenate(groundtruth_classes, axis=0)
        groundtruth_valids = np.concatenate(groundtruth_valids, axis=0)

        self.coco = CocoEvalidation(groundtruth_boxes,groundtruth_classes,groundtruth_valids,class_names)


    def eval(self):
        """Get mAP value from CoCo Tool
        """
        detection_boxes = []; detection_scores = []; detection_classes = []; detection_valids = []
        pred_generator_tqdm = tqdm(self.pred_generator, total=len(self.pred_generator))
        for batch_img, _,_ in pred_generator_tqdm:

            model_outputs = self.model.predict(batch_img)
            pre_nms_decoded_boxes, pre_nms_scores = postprocess(model_outputs, self.args)
            pre_nms_decoded_boxes = pre_nms_decoded_boxes.numpy()
            pre_nms_scores = pre_nms_scores.numpy()

            boxes, scores, classes, valid_detections = NonMaxSuppression.diou_nms_np(pre_nms_decoded_boxes, pre_nms_scores, self.args)

            detection_boxes.append(boxes)
            detection_scores.append(scores)
            detection_classes.append(classes)
            detection_valids.append(valid_detections)
            pred_generator_tqdm.set_description("Evaluation...")

        detection_boxes = np.concatenate(detection_boxes, axis=0)
        detection_scores = np.concatenate(detection_scores, axis=0)
        detection_classes = np.concatenate(detection_classes, axis=0)
        detection_valids = np.concatenate(detection_valids, axis=0)

        return self.coco.get_coco_mAP(detection_boxes, detection_scores, detection_classes, detection_valids)