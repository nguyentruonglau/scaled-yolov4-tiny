from PIL import Image
import tensorflow as tf

from generator.data_augmentation import alb_augmentation
from generator.anchor_generator import anchor_generator
from generator.anchor_generator import anchor_generator_one_class
from utility.preprocess import preprocess_data

import os
import time
import warnings
import numpy as np
import cv2
import copy

from utility.preprocess import resize_img, resize_boxes
from utility.preprocess import normalize_boxes
import xml.etree.ElementTree as ET
from generator import config_data as cfg


def data_generator(args):
    """Get training and validation data
    """
    train_dataset = Generator(args, mode=0)
    valid_dataset = Generator(args, mode=1)
    return train_dataset, valid_dataset


class Generator(tf.keras.utils.Sequence):
    def __init__(self, args, mode=0):
        #general
        self._args = args
        self.mode = mode
        
        self.batch_size = self._args.batch_size
        self.skip_difficult = int(self._args.skip_difficult)
        self.model_shape = self._args.model_shape
        self.augment = True if self.mode == 0 else False

        #contain all paths
        self.xml_path_list = []
        self.img_path_list = []
        self.boxes_and_labels = []
        self.class_names_dict={}

        with open(self._args.class_names) as f:
            self.class_names = f.read().splitlines()
            for i in range(len(self.class_names)):
                self.class_names_dict[self.class_names[i]] = i
        root_dir = self._args.dataset

        #for read train/val set
        if self.mode == 0: txt_path = self._args.train_set
        else: txt_path = self._args.val_set

        #open train/set set
        with open(txt_path) as f:
            lines = f.readlines()
        for line in lines:
            valid_label = self.checking_annotation_file(
                os.path.join(root_dir, 'Annotations', line.strip()+'.xml')
            )
            if valid_label:
                self.xml_path_list.append(os.path.join(root_dir, 'Annotations', line.strip() + '.xml'))
                self.img_path_list.append(os.path.join(root_dir, 'JPEGImages', line.strip() + '.jpg'))

        for xml_path in self.xml_path_list:
            # boxes_and_labels -> [M,N,4], M images, N bboxs/image, 4 coordinates
            self.boxes_and_labels.append(self.parse_xml(xml_path))

        #indexing the dataset
        self.data_index = np.empty([len(self.img_path_list)], np.int32)
        for index in range(len(self.img_path_list)):
            self.data_index[index] = index

    def checking_annotation_file(self, xml_path):
        try:
            tree = ET.parse(xml_path)
            xml_root = tree.getroot()
            num_valid_boxes = 0
            for element in xml_root.iter('object'):
                difficult = int(element.find('difficult').text)
                if difficult:
                    continue
                num_valid_boxes += 1
            if num_valid_boxes == 0:
                return False
        except:
            return False
        return True

    def value_box(box):
        if np.abs(box[2]-box[0])<2 or np.abs(box[3]-box[1])<2:
            return False
        return True

    def parse_xml(self,file_path):
        try:
            tree = ET.parse(file_path)
            xml_root = tree.getroot()

            size = xml_root.find('size')
            width = float(size.find('width').text)
            height = float(size.find('height').text)

            boxes = np.empty((len(xml_root.findall('object')), 5))
            box_index = 0
            for i, element in enumerate(xml_root.iter('object')):
                difficult = int(element.find('difficult').text)
                class_name = element.find('name').text
                box = np.zeros((4,))
                label = self.class_names_dict[class_name]
                bndbox = element.find('bndbox')

                box[0] = float(bndbox.find('xmin').text)-1
                box[1] = float(bndbox.find('ymin').text)-1
                box[2] = float(bndbox.find('xmax').text)-1
                box[3] = float(bndbox.find('ymax').text)-1

                box[0] = np.maximum(box[0], 0)
                box[1] = np.maximum(box[1], 0)
                box[2] = np.minimum(box[2], width-1)
                box[3] = np.minimum(box[3], height-1)

                if (difficult and self.skip_difficult) or not self.value_box:
                    continue
                box = normalize_boxes(box)
                if not self.value_box:
                    continue

                boxes[box_index, 0:4] = box
                boxes[box_index, 4] = int(label)
                box_index += 1
            return boxes[0:box_index] 
        except ET.ParseError as e:
            ValueError('[ERROR]: Parsing xml file: {}: {}'.format(file_path, e))

    def apply_data_augmentation(self, image, boxes):
        """Apply data augmentation
        """
        #make copy
        tmp_image = copy.deepcopy(image)
        tmp_boxes = copy.deepcopy(boxes)

        #albumentations, use RGB image
        class_labels = np.array(boxes[:,4], dtype=np.uint8)
        boxes = np.array(boxes[:,0:4], dtype=np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image, boxes, class_labels = alb_augmentation(image, boxes, class_labels)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if len(boxes) == 0: return tmp_image, tmp_boxes

        #boxes = [xmin,ymin,xmax,ymax,label]
        boxes = np.array(boxes, dtype=np.float32)
        class_labels = np.array(class_labels, dtype=np.uint8)
        boxes = np.hstack((boxes, np.expand_dims(class_labels, axis=1)))
        return image, boxes

    def get_classes_num(self):
        return int(self._args.num_classes)

    def get_size(self):
        return len(self.img_path_list)

    def __len__(self):
        if self.mode == 0: return len(self.img_path_list) // self.batch_size
        else: return int(np.ceil(len(self.img_path_list) / self.batch_size))

    def on_epoch_end(self):
        if self.mode == 0: np.random.shuffle(self.data_index)

    def __getitem__(self, item):
        with tf.device("/cpu:0"):
            groundtruth_valids = np.zeros([self.batch_size], np.int)#contain number boxes per image
            self.max_side = self.min_side = self.model_shape

            # get images and bounding boxes
            batch_img = np.zeros([self.batch_size, self.min_side, self.max_side, 3])
            batch_boxes = np.empty([self.batch_size, self._args.max_box_num_per_image, 5])
            batch_boxes_list = []

            # traverse each batch of data
            for batch_index, file_index in enumerate(self.data_index[item*self.batch_size:(item+1)*self.batch_size]):
                #get image from file
                img = cv2.imread(self.img_path_list[file_index])
                img, scale, pad = resize_img(img, (self.min_side, self.max_side), 0)

                #fill image
                batch_img[batch_index, 0:img.shape[0], 0:img.shape[1], :] = img

                #fill boxes
                boxes = self.boxes_and_labels[file_index]
                boxes = resize_boxes(boxes, scale, pad)
                batch_boxes_list.append(boxes)

                #number of boxes per image
                groundtruth_valids[batch_index] = boxes.shape[0]
                boxes = np.pad(
                    boxes, 
                    [(0, self._args.max_box_num_per_image-boxes.shape[0]), (0, 0)], 
                    mode='constant' 
                )
                batch_boxes[batch_index] = boxes
            tail_batch_size = len(batch_boxes_list)

            #augment data
            if self.augment:
                for idx in range(self.batch_size):

                    image = np.array(batch_img[idx], dtype=np.uint8)
                    boxes = np.array(batch_boxes_list[idx], dtype=np.float32)

                    image, boxes = self.apply_data_augmentation(image, boxes)

                    #update batch_boxes, batch_img, number of boxes
                    groundtruth_valids[idx] = boxes.shape[0]
                    boxes = np.pad(boxes,[(0, self._args.max_box_num_per_image-boxes.shape[0]), (0, 0)], mode='constant')
                    batch_boxes[idx] = boxes
                    batch_img[idx] = image

            batch_img = batch_img[0:tail_batch_size]
            batch_boxes = batch_boxes[0:tail_batch_size]
            groundtruth_valids = groundtruth_valids[0:tail_batch_size]

            batch_img, batch_boxes = preprocess_data(batch_img, batch_boxes)
            if int(self._args.num_classes) == 1:
                y_true = anchor_generator_one_class(
                    self.max_side, batch_boxes, 
                    groundtruth_valids, 
                    self._args
                )
            else:
                y_true = anchor_generator(
                    self.max_side, 
                    batch_boxes, 
                    groundtruth_valids, 
                    self._args
                )

            if self.mode == 1:
                return batch_img, batch_boxes, groundtruth_valids
            return batch_img, y_true
            return batch_img, y_true, batch_boxes