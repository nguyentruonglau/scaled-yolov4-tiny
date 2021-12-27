from copy import deepcopy
import os, cv2, time
import logging
import warnings
import numpy as np
from PIL import Image


def boxes_iou(boxes1, boxes2):
    """Matched
    Args:
        Boxes1 [3D array]: [batch_size, args.num_boxes, 5]
        Boxes2 [2D]: [[W, H]]
    Returns:
        Intersection of union
    """
    boxes2 = np.array(boxes2)
    boxes1 = np.expand_dims(boxes1, -2)
    boxes1_wh = boxes1[..., 2:4] - boxes1[..., 0:2]
    inter_area = np.minimum(boxes1_wh[..., 0], boxes2[..., 0]) * np.minimum(boxes1_wh[..., 1], boxes2[..., 1])
    boxes1_area = boxes1_wh[..., 0] * boxes1_wh[..., 1]
    boxes2_area = boxes2[..., 0] * boxes2[..., 1]
    return inter_area/(boxes1_area+boxes2_area-inter_area)


def preprocess_data(img, boxes, style=0):
    """Normalize data before entering the model
    """
    if style == 0: img = img/255.0
    elif style == 1: img = img/127.5-1.0
    boxes[..., 0:4] /= np.tile(img[0].shape[0:2][::-1], [2])
    return img.copy(), boxes.copy()


def normalize_boxes(box):
    """Normalize the coordinate values of bounding boxes to [xmin,ymin,xmax,ymax]
    """
    tmp_box = deepcopy(box)
    box[0] = np.minimum(tmp_box[0], tmp_box[2])
    box[2] = np.maximum(tmp_box[0], tmp_box[2])
    box[1] = np.minimum(tmp_box[1], tmp_box[3])
    box[3] = np.maximum(tmp_box[1], tmp_box[3])
    return box


def resize_boxes(boxes, scale, pad_size):
    """Resize boxes by scale and pad_size when resizing images
    """
    boxes = np.array(boxes, dtype=float)
    boxes[:, 0:4] *= scale
    half_pad = pad_size // 2
    boxes[:, 0:4] += np.tile(half_pad,2)
    return boxes


def resize_img(img, dst_size, constant_value):
    """Resize image with default interpolation (cv2.INTER_LINEAR)
    """
    scale = np.minimum(dst_size[0]/img.shape[1], dst_size[1]/img.shape[0])
    img = cv2.resize(img, None, fx=scale, fy=scale)

    pad_size = np.array(dst_size, dtype=int) - np.array(img.shape[0:2][::-1], dtype=int)
    half_pad_size = np.array((pad_size//2), dtype=int)
    img = np.pad(img,
        [
            (half_pad_size[1],pad_size[1]-half_pad_size[1]),
            (half_pad_size[0],pad_size[0]-half_pad_size[0]),
            (0, 0)
        ],
        constant_values=constant_value
    )
    return img, scale, pad_size


def decode_boxes(boxes, input_shape, pad_size, scale):
    """Convert boxes to img_ori coordination
    """
    boxes = np.array(boxes, dtype=float)
    boxes = boxes * np.tile(input_shape, 2)
    boxes = boxes - np.tile(pad_size, 2)//2
    boxes = np.array(boxes/scale, dtype=int)
    return boxes


def iou(box, clusters):
    """Calculates the Intersection over Union (IoU) between a box and k clusters.
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_


def avg_iou(boxes, clusters):
    """Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def kmeans(boxes, k, dist=np.median):
    """Calculates k-means clustering with the Intersection over Union (IoU) metric.
    """
    rows = boxes.shape[0]
    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))
    np.random.seed()
    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]
    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
        nearest_clusters = np.argmin(distances, axis=1)
        if (last_clusters == nearest_clusters).all():
            break
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters==cluster], axis=0)
        last_clusters = nearest_clusters
    return clusters