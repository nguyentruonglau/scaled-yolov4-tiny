import os,sys; sys.path.append('..')

from imutils.paths import list_images
from imutils.paths import list_files
import matplotlib.pyplot as plt

from lxml import etree
from random import shuffle
from tqdm import tqdm
from PIL import Image
import numpy as np

import argparse
import sys


def parse_args(args):
    parser = argparse.ArgumentParser("")
    parser.add_argument('--annotation-dir', type=str, default='../database/dataset/Car/Annotations')
    parser.add_argument('--test-size', type=float, default=0.1, help='Percentage of data used for testing')
    parser.add_argument('--input-data-dir', type=str, default='../database/dataset/Car/JPEGImages', 
                        help='path to directory contain images')
    parser.add_argument('--class-names', default='../database/dataset/Car/ClassNames/car.names')
    parser.add_argument('--output-dir', type=str, default='../database/dataset/Car/ImageSets')
    return parser.parse_args(args)


def check_parser(args):
    """Check parse_args is valid or not
    """
    if not os.path.exists(args.input_data_dir):
        raise ValueError('[ERROR]: {} not found'.format(args.input_data_dir))
    if not os.path.exists(args.annotation_dir):
        raise ValueError('[ERROR]: {} not found'.format(args.annotation_dir))
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)


def remove_images(image_paths, annotation_paths):
    """Remove images not found annotaions files
    """
    for img_path in copy.deepcopy(image_paths):
        FLAGS = False
        for anno_path in annotation_paths:
            if clean_path(os.path.basename(img_path)[0:-4]) == clean_path(os.path.basename(anno_path)[0:-4]):
                FLAGS = True
                break
        if FLAGS == False:
            try:
                os.remove(img_path)
            except OSError as e:
                print("[ERROR]: Failed with: ", e.strerror)
            image_paths.remove(img_path)
    return image_paths


def remove_annotations(image_paths, annotation_paths):
    """Remove annotations not found image files
    """
    for anno_path in copy.deepcopy(annotation_paths):
        FLAGS = False
        for img_path in image_paths:
            if clean_path(os.path.basename(anno_path)[0:-4]) == clean_path(os.path.basename(img_path)[0:-4]):
                FLAGS = True
                break
        if FLAGS == False:
            try:
                os.remove(anno_path)
            except OSError as e:
                print("[ERROR]: Failed with: ", e.strerror)
            annotation_paths.remove(anno_path)
    return annotation_paths


def clean_path(path):
    """Clean the path
    """
    clean_path = ''; tmp_path = path.split(' ')
    for item in tmp_path:
        clean_path += item
    return clean_path


def check_matched(image_paths, annotation_paths):
    """Check if the image files and annotation files match?

        The function will clean the data includeing: 
            + Delete images without labels, 
            + Delete labels without images. 
    """
    print('[INFOR]: Check matching and file format...')
    #check if the image is JPG or not
    for img_path in image_paths:
        if not img_path.endswith(".jpg"): raise ValueError('[ERROR]: Images must .jpg')

    #check if the annotation is XML or not
    for ant_path in annotation_paths:
        if not ant_path.endswith(".xml"): raise ValueError('[ERROR]: Annotations must .xml')

    #sort up ascending
    image_paths = list(np.sort(image_paths))
    annotation_paths = list(np.sort(annotation_paths))

    #remove image have not annotation
    remove_images(image_paths, annotation_paths)
    #remove annotation have not image
    remove_annotations(image_paths, annotation_paths)

    print('[INFOR]: Images and annotations matched')
    return image_paths, annotation_paths


def recursive_parse_xml_to_dict(xml):
    """Recursively parses XML contents to python dict.
    We assume that `object` tags are the only ones that can appear
    multiple times at the same level of a tree.

    Args:
        xml: xml tree obtained by parsing XML file contents using lxml.etree
    Returns:
        Python dictionary holding XML contents.
    """
    # if don't see, exit
    if len(xml) == 0:
        return {xml.tag: xml.text}
    result = {}
    # else continue recursive
    for child in xml:
        child_result = recursive_parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def get_annotation_infor(input_annotation_path):
    #get infor xml file
    with open(input_annotation_path, 'rb') as file:
        xml_str = file.read()
    xml = etree.fromstring(xml_str)
    infors = recursive_parse_xml_to_dict(xml)['annotation']

    #get bbox for drawing
    objectnames = []
    if 'object' in infors:
        for obj in infors['object']:
            objectnames.append(obj['name'].lower())
    return objectnames


def split_dataset(args, annotation_paths, class_names):
    """Divide the data so that the data ratio of the classes is the same
    """
    FLAGS = False
    train_annotation_paths, val_annotation_paths = None, None

    while FLAGS == False:
        #shuffle image paths
        shuffle(annotation_paths)

        #split train/test
        num_train = int(len(annotation_paths) - len(annotation_paths)*args.test_size)
        train_annotation_paths = annotation_paths[0:num_train]
        val_annotation_paths = annotation_paths[num_train:-1]

        #count number of object per class
        val_data = dict(); train_data = dict()
        for class_name in class_names:
            val_data[class_name] = 0
            train_data[class_name] = 0

        #get object train images
        for train_path in train_annotation_paths:
            objectnames = get_annotation_infor(train_path)
            for ob_name in objectnames: train_data[ob_name] += 1

        #get object val images
        for val_path in val_annotation_paths:
            objectnames = get_annotation_infor(val_path)
            for ob_name in objectnames: val_data[ob_name] += 1

        #split dataset by test_size scale
        for class_name in class_names:
            if val_data[class_name] / (train_data[class_name] + 1e-3) < args.test_size:
                FLAGS = False
                break
            FLAGS = True
    #statistic
    print('\n[INFOR]: STATISTICS:')
    for class_name in class_names:
        print('{}:'.format(class_name))
        print('train: {}'.format(train_data[class_name]))
        print('val: {}\n'.format(val_data[class_name]))

    draw_statistics(class_names, train_data, val_data)
    print('[INFOR] See more result in ./database/logdata/')
    return train_annotation_paths, val_annotation_paths


def draw_statistics(class_names, train_data, val_data):
    locations = np.arange(len(class_names))
    width_bar = 0.3

    fig = plt.figure()
    ax = fig.add_subplot(111)

    #get value
    train = []; val = []
    for key, value in train_data.items(): train.append(value)
    for key, value in val_data.items(): val.append(value)

    rects1 = ax.bar(locations, val, width_bar, color='#bb51f0')
    rects2 = ax.bar(locations+width_bar, train, width_bar, color='#7769f5')

    ax.set_ylabel('Bounding Boxes')
    ax.set_xticks(locations + width_bar)
    ax.set_xticklabels(class_names, rotation=90)
    ax.legend((rects1[0], rects2[0]), ('val', 'train'))

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                    ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)
    plt.savefig('../database/logdata/statistic.png')


def main(args):
    #load class name of dataset
    with open(args.class_names) as f:
        class_names = f.read().splitlines()

    #annotation paths
    annotation_paths = list(list_files(args.annotation_dir))
    if len(annotation_paths) == 0:
        raise ValueError(
            '[ERROR]: XML annotation files in {} not found'.format(args.annotation_dir)
        )

    #image paths
    image_paths = list(list_images(args.input_data_dir))
    if len(image_paths) == 0:
        raise ValueError(
            '[ERROR]: Images in {} not found'.format(args.input_data_dir)
        )

    #check matched
    image_paths, annotation_paths = check_matched(image_paths, annotation_paths)
        
    #auto split dataset
    train_image_paths, val_image_paths = split_dataset(
        args, 
        annotation_paths, 
        class_names
    )

    print('[INFOR]: Start writing!')
    #create train.txt
    print('[INFOR]: Number files of training: {}'.format(len(train_image_paths)))
    with open(os.path.join(args.output_dir, 'train.txt'), 'w') as file:
        for i in tqdm(range(len(train_image_paths))):
            train_image_name = os.path.basename(train_image_paths[i])[:-4]
            file.write(train_image_name+'\n')

    #create val.txt
    print('[INFOR]: Number files of validation: {}'.format(len(val_image_paths)))
    with open(os.path.join(args.output_dir, 'val.txt'), 'w') as file:
        for i in tqdm(range(len(val_image_paths))):
            train_image_name = os.path.basename(val_image_paths[i])[:-4]
            file.write(train_image_name+'\n')

    print('[INFO]: Complete!')


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    print('output_dir=',args.output_dir)
    print('test_size=',args.test_size)
    print('input_data_dir=',args.input_data_dir)
    print('annotation_dir=',args.annotation_dir)
    print('class_names=',args.class_names)

    main(args)