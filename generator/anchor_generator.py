from PIL import Image
from generator.config_data import yolo_anchors
from utility.preprocess import boxes_iou

import argparse
import os
import logging
import time
import warnings
import numpy as np
import cv2


def anchor_generator(max_side, batch_boxes, groundtruth_valids, args):
    #variable initialization
    strides = [16 * 2 ** i for i in range(2)]
    offset = 0.5
    class_num = int(args.num_classes)
    batch_size = batch_boxes.shape[0]
    grid_size = max_side//np.array(strides).astype(np.int32)

    iou_scores = boxes_iou(batch_boxes, np.reshape(np.array(yolo_anchors), [-1,2])/(max_side, max_side))
    #matched_anchor_index: [batch_size, args.num_boxes, len(yolo_anchors)]
    matched_anchor_index = np.argsort(-iou_scores, axis=-1)
    #matched_anchor_num (number): [batch_size, args.num_boxes]
    matched_anchor_num = np.sum(iou_scores>args.anchor_match_iou_thr, axis=-1)
    if args.anchor_match_iou_thr == -1: matched_anchor_num = np.ones_like(matched_anchor_num)

    matched_anchor_num = np.expand_dims(matched_anchor_num, axis=-1)
    #batch_boxes: [batch_size, args.num_boxes, 5+len(akaDET_anchors)+1]
    batch_boxes = np.concatenate([batch_boxes, matched_anchor_index, matched_anchor_num], axis=-1)

    grids_0 = np.zeros([batch_size, grid_size[0], grid_size[0], len(yolo_anchors[0]), 5 + class_num], np.float32)
    grids_1 = np.zeros([batch_size, grid_size[1], grid_size[1], len(yolo_anchors[0]), 5 + class_num], np.float32)
    grids = [grids_0, grids_1]

    for batch_index in range(batch_size):
        for box_index in range(groundtruth_valids[batch_index]):
            for anchor_index in batch_boxes[batch_index][box_index][5:5 + int(batch_boxes[batch_index][box_index][-1])]:
            	#if anchor_index > len(akaDET_anchors) -> small boxes move to min scale
                grid_index = (anchor_index // len(yolo_anchors[0])).astype(np.int32)
                grid_anchor_index = (anchor_index % len(yolo_anchors[0])).astype(np.int32)
                
                cxcy = (batch_boxes[batch_index][box_index][0:2]+batch_boxes[batch_index][box_index][2:4])/2
                cxcy *= grid_size[grid_index]
                grid_xy = np.floor(cxcy).astype(np.int32)
                dxdy = cxcy - grid_xy
                dwdh = batch_boxes[batch_index][box_index][2:4]-batch_boxes[batch_index][box_index][0:2]
                dwdh = dwdh*np.array(grid_size[grid_index])
                grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]][grid_anchor_index][4] = 1
                grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]][grid_anchor_index][5+batch_boxes[batch_index][box_index][4].astype(np.int32)] = 1
                
                grid_xy_fract = cxcy%1.
                if (grid_xy > 0).all():
                    if grid_xy_fract[0] < offset:
                        dxdy = cxcy - np.floor(cxcy - [0.5, 0.])
                        grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]-1][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                        grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]-1][grid_anchor_index][4] = 1
                        grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]-1][grid_anchor_index][
                            5 + batch_boxes[batch_index][box_index][4].astype(np.int32)] = 1

                    if grid_xy_fract[1] < offset:
                        dxdy = cxcy - np.floor(cxcy - [0., 0.5])
                        grids[grid_index][batch_index][grid_xy[1]-1][grid_xy[0]][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                        grids[grid_index][batch_index][grid_xy[1]-1][grid_xy[0]][grid_anchor_index][4] = 1
                        grids[grid_index][batch_index][grid_xy[1]-1][grid_xy[0]][grid_anchor_index][
                            5 + batch_boxes[batch_index][box_index][4].astype(np.int32)] = 1

                if (grid_xy<grid_size[grid_index]-1).all():
                    if grid_xy_fract[0] > offset:
                        dxdy = cxcy - np.floor(cxcy + [0.5, 0.])
                        grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]+1][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                        grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]+1][grid_anchor_index][4] = 1
                        grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]+1][grid_anchor_index][
                            5 + batch_boxes[batch_index][box_index][4].astype(np.int32)] = 1
                    if grid_xy_fract[1] > offset:
                        dxdy = cxcy - np.floor(cxcy + [0., 0.5])
                        grids[grid_index][batch_index][grid_xy[1]+1][grid_xy[0]][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                        grids[grid_index][batch_index][grid_xy[1]+1][grid_xy[0]][grid_anchor_index][4] = 1
                        grids[grid_index][batch_index][grid_xy[1]+1][grid_xy[0]][grid_anchor_index][
                            5 + batch_boxes[batch_index][box_index][4].astype(np.int32)] = 1
    return tuple(grids)


def anchor_generator_one_class(max_side, batch_boxes, groundtruth_valids, args):
    #variable initialization
    strides = [16 * 2 ** i for i in range(2)]
    offset = 0.5; class_num = 0
    batch_size = batch_boxes.shape[0]
    grid_size = max_side//np.array(strides).astype(np.int32)

    iou_scores = boxes_iou(batch_boxes, np.reshape(np.array(yolo_anchors), [-1,2])/(max_side, max_side))
    #matched_anchor_index: [batch_size, args.num_boxes, len(yolo_anchors)]
    matched_anchor_index = np.argsort(-iou_scores,axis=-1)
    #matched_anchor_num (number): [batch_size, args.num_boxes]
    matched_anchor_num = np.sum(iou_scores>args.anchor_match_iou_thr, axis=-1)
    if args.anchor_match_iou_thr == -1: matched_anchor_num = np.ones_like(matched_anchor_num)

    matched_anchor_num = np.expand_dims(matched_anchor_num, axis=-1)
    #batch_boxes: [batch_size, args.num_boxes, 5+len(akaDET_anchors)+1]
    batch_boxes = np.concatenate([batch_boxes, matched_anchor_index,matched_anchor_num], axis=-1)

    grids_0 = np.zeros([batch_size, grid_size[0], grid_size[0], len(yolo_anchors[0]), 5 + class_num], np.float32)
    grids_1 = np.zeros([batch_size, grid_size[1], grid_size[1], len(yolo_anchors[0]), 5 + class_num], np.float32)
    grids = [grids_0, grids_1]

    for batch_index in range(batch_size):
        for box_index in range(groundtruth_valids[batch_index]):
            for anchor_index in batch_boxes[batch_index][box_index][5:5 + int(batch_boxes[batch_index][box_index][-1])]:
                #if anchor_index > len(akaDET_anchors) -> small boxes move to min scale
                grid_index = (anchor_index // len(yolo_anchors[0])).astype(np.int32)
                grid_anchor_index = (anchor_index % len(yolo_anchors[0])).astype(np.int32)
                cxcy = (batch_boxes[batch_index][box_index][0:2]+batch_boxes[batch_index][box_index][2:4])/2
                cxcy *= grid_size[grid_index]
                grid_xy = np.floor(cxcy).astype(np.int32)

                dxdy = cxcy - grid_xy
                dwdh = batch_boxes[batch_index][box_index][2:4]-batch_boxes[batch_index][box_index][0:2]
                dwdh = dwdh*np.array(grid_size[grid_index])
                grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]][grid_anchor_index][4] = 1
                grid_xy_fract = cxcy%1.
                if (grid_xy > 0).all():
                    if grid_xy_fract[0] < offset:
                        dxdy = cxcy - np.floor(cxcy - [0.5, 0.])
                        grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]-1][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                        grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]-1][grid_anchor_index][4] = 1

                    if grid_xy_fract[1] < offset:
                        dxdy = cxcy - np.floor(cxcy - [0., 0.5])
                        grids[grid_index][batch_index][grid_xy[1]-1][grid_xy[0]][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                        grids[grid_index][batch_index][grid_xy[1]-1][grid_xy[0]][grid_anchor_index][4] = 1

                if (grid_xy<grid_size[grid_index]-1).all():
                    if grid_xy_fract[0] > offset:
                        dxdy = cxcy - np.floor(cxcy + [0.5, 0.])
                        grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]+1][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                        grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]+1][grid_anchor_index][4] = 1

                    if grid_xy_fract[1] > offset:
                        dxdy = cxcy - np.floor(cxcy + [0., 0.5])
                        grids[grid_index][batch_index][grid_xy[1]+1][grid_xy[0]][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                        grids[grid_index][batch_index][grid_xy[1]+1][grid_xy[0]][grid_anchor_index][4] = 1

    return tuple(grids)