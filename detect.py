from utils import test_time_aug
from utility.preprocess import decode_boxes
from utility.preprocess import resize_img

import random, cv2, time, argparse
import os, sys, warnings

import tensorflow as tf
import onnxruntime as rt
import numpy as np

from tqdm import tqdm
from copy import deepcopy


# enable xla devices
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# re-writes the environment variables and makes only certain NVIDIA GPU(s) visible for that process.
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #one GPU used, have id=0

# dynamic allocate GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices: tf.config.experimental.set_memory_growth(physical_devices[0], True)
else: warnings.warn('[WARNING]: GPU not found, CPU current is being used')


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--class-names', default='./database/dataset/Car/ClassNames/car.names')
    parser.add_argument('--input-data',default='./database/images/input')
    parser.add_argument('--output-data', default='./database/images/output', type=str, help='Directory will contain predicted images')

    parser.add_argument('--model-path', default='./database/models/onnx/carmodels416.onnx')
    parser.add_argument('--input-shape', default=(416, 416), help='Input shape of model')

    parser.add_argument('--nms-max-box-num', default=100, type=int, help='Max number of bounding boxes per image')
    parser.add_argument('--nms-iou-threshold', default=0.2, type=float, help='Intersection of union threshold')
    parser.add_argument('--nms-score-threshold', default=0.4, type=float, help='Score~Confidence threshold')

    parser.add_argument('--machine-used', default='CPU', help="choices = ['CPU', 'GPU']")
    return parser.parse_args(args)


def draw_bbox(args, img, bboxs, scores, object_names, class_colors):
    """Draw bounding boxes for an image
    """
    # adjust to fit the size of the frame
    scale = max(img.shape[0:2]) / 416
    line_width = int(2 * scale)
    font = cv2.FONT_HERSHEY_DUPLEX

    for object_name, bbox, score in zip(object_names, bboxs, scores):
        if score < args.nms_score_threshold:
            continue
        color = class_colors[object_name]
        label_boxes = object_name + "," + str("%0.2f" % score)
        # get coordinates of bounding box
        x1, y1, x2, y2 = bbox
        # draw bounding
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)
        font_scale = max(0.3 * scale, 0.3)
        thickness = max(int(1 * scale), 1)
        (text_width, text_height) = cv2.getTextSize(
                    label_boxes, font, fontScale=font_scale, thickness=thickness)[0]
        cv2.rectangle(img, (x1 - line_width//2, y1 - text_height),
                (x1 + text_width, y1), color, cv2.FILLED)
        cv2.putText(img, label_boxes, (x1, y1),
                font, font_scale,
                (255, 255, 255), 
                thickness, cv2.LINE_AA)
    return img


def main(args):
    # load model, and choose device in providers
    providers = "CPUExecutionProvider" if args.machine_used == "CPU" else "CUDAExecutionProvider"
    model = rt.InferenceSession(args.model_path, providers=[providers])
    input_name = model.get_inputs()[0].name
    label_boxes = model.get_outputs()[0].name
    label_scores = model.get_outputs()[1].name

    # load class name of dataset
    with open(args.class_names) as f:
        class_names = f.read().splitlines()
    img_list = np.array(os.listdir(args.input_data))

    # generate colors
    class_colors = {name: list(np.random.random(size=3)*255) for name in class_names}

    # start detection
    for i in tqdm(range(len(img_list))):
        # read one image
        img_ori = cv2.imread(os.path.join(args.input_data, img_list[i]))
        img = deepcopy(img_ori)

        # resize image without distortion
        img, scale, pad_size = resize_img(img, args.input_shape, 0)
        img = np.array(img, dtype=np.float32)

        # object detection on image
        bboxs, scores, classes = test_time_aug(args, img, model, input_name, label_boxes, label_scores)

        bboxs = decode_boxes(bboxs, args.input_shape, pad_size, scale)
        # plot bouding boxs
        object_names = [class_names[i] for i in classes]
        img_ori= draw_bbox(args, img_ori, bboxs, scores, object_names, class_colors)
        # write predicted image
        cv2.imwrite(os.path.join(args.output_data, img_list[i]), img_ori)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)