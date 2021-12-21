from threading import Thread
from queue import Queue
import tensorflow as tf
import onnxruntime as rt
from imutils.paths import list_files
import numpy as np

from utility.preprocess import decode_boxes
from utility.preprocess import resize_img

import warnings
import random
import os
import cv2
import time
import argparse

from model.nonmaxsuppression import NonMaxSuppression

#providing access to the CPU while the GPU is executing
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

#re-writes the environment variables and makes only certain NVIDIA GPU(s) visible for that process.
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #one GPU used, have id=0

#dynamic allocate GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices: tf.config.experimental.set_memory_growth(physical_devices[0], True)
else: 
    warnings.warn('[WARNING]: GPU not found, CPU current is being used')


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class-names', default='./database/dataset/Car/ClassNames/car.names')
    parser.add_argument("--input-data", type=str, default='./database/videos/input/video.mp4',help="Video source. If empty, uses webcam 0 stream")
    parser.add_argument("--filename", type=str, default='./database/videos/output/video.avi',help="Inference video name. Not saved if empty")
    parser.add_argument("--dont-show", default=False, help="Windown inference display")

    parser.add_argument("--input-shape", type=tuple, default=(1280, 720), help="Shape of video/stream camera") #WxH
    parser.add_argument("--model-shape", type=tuple, default=(416, 416), help='Input shape of model')
    parser.add_argument("--model-path", type=str, default='./database/models/onnx/carmodels416.onnx')

    parser.add_argument('--nms-max-box-num', default=100, type=int, help='Max number of bounding boxes per image')
    parser.add_argument('--nms-iou-threshold', default=0.2, type=float, help='Intersection of union threshold')
    parser.add_argument('--nms-score-threshold', default=0.4, type=float, help="Remove detections with confidence below this value")

    parser.add_argument('--machine-used', default='CPU', help="choices = ['CPU', 'GPU']")
    return parser.parse_args()


def str2int(video_path):
    """
    Argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def check_arguments_errors(args):
    assert 0 < args.nms_score_threshold < 1, "[ERROR]: Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.model_path):
        raise ValueError("[ERROR]: Invalid model_path {}".format(os.path.abspath(args.model_path)))

    if str2int(args.input_data) == str and not os.path.exists(args.input_data):
        raise ValueError("[ERROR]: Invalid video path {}".format(os.path.abspath(args.input_data)))


def video_capture(frame_queue, akadet_image_queue):
    while cap.isOpened():

        ret, frame_org = cap.read()
        if not ret:
            break
        #frame for draw bbox
        frame_queue.put(frame_org)

        #preprocessing image
        frame, scale, pad_size = resize_img(frame_org, args.model_shape, 0)
        frame = np.expand_dims(frame/255.0, axis=0)
        frame = np.array(frame, dtype=np.float32)

        #frame for detect
        akadet_image_queue.put([frame, scale, pad_size])

    cap.release()


def inference(akadet_image_queue, detections_queue, fps_queue):
    while cap.isOpened():
        image, scale, pad_size = akadet_image_queue.get()

        prev_time = time.time()
        # Get predict of model
        pre_nms_decoded_boxes, pre_nms__scores = model.run([label_boxes, label_scores], {input_name: image})

        # Non-maximum suppression
        bbox, confidence, classes, _ = NonMaxSuppression.diou_nms_np(pre_nms_decoded_boxes, pre_nms__scores, args)

        # Detections_queue have only one element
        detections_queue.put((classes, confidence, bbox, scale, pad_size))

        # Get FPS
        fps = int(1/(time.time() - prev_time + 1e-9))
        fps_queue.put(fps)
        print('FPS: {}'.format(fps))

    cap.release()


def draw_bbox(img, detections, random_color=True, show_text=True):
    #get detections
    classes, confidence, bbox, scale, pad_size = detections
    #decode bouding boxes
    bbox = decode_boxes(bbox[0], args.model_shape, pad_size, scale)

    for _class, _confidence, _bbox in zip(classes[0], confidence[0], bbox):
        #no plot box for object that have confidence < thresh
        if _confidence < args.nms_score_threshold:
            continue
        #get label
        _label = str(class_names[int(_class)])
        
        #get coordinates of bounding box
        x1, y1, x2, y2 = _bbox

        #_label: people, the people class will have its own color to
        #distinguish it from other classes
        color = class_colors[_label]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        if show_text:
            text = f'{_label} {_confidence:.2f}'
            font = cv2.FONT_HERSHEY_DUPLEX
            (text_width, text_height) = cv2.getTextSize(
                    text, font, fontScale=0.3, thickness=1
                )[0]
            cv2.rectangle(img, (x1, y1 - text_height),
                (x1 + text_width, y1),
                color, cv2.FILLED)
            cv2.putText(img, text, (x1, y1),
                font, 0.3, (255, 255, 255),
                1, cv2.LINE_AA)
    return img


def drawing(frame_queue, detections_queue, fps_queue):
    random.seed(3)  # deterministic bbox colors
    video = set_saved_video(cap, args.filename, args.input_shape)
    while cap.isOpened():

        frame = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        # draw bbox for each image
        if frame is not None:
            image = draw_bbox(frame, detections) #BGR
            if not args.dont_show:
                cv2.imshow('INFERENCE', image)
            if args.filename is not None:
                video.write(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    video.release()
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    # define queues
    frame_queue = Queue()
    akadet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

    #get model
    args = parser()
    check_arguments_errors(args)

    #get class_name
    with open(args.class_names) as f:
        class_names = f.read().splitlines()
    print('[INFOR]: Load class names file success.')

    #generate colors
    class_colors = {name: list(np.random.random(size=3)*255) for name in class_names}

    #load model
    providers = "CPUExecutionProvider" if args.machine_used == "CPU" else "CUDAExecutionProvider"
    model = rt.InferenceSession(args.model_path, providers=[providers])
    input_name = model.get_inputs()[0].name
    label_boxes = model.get_outputs()[0].name
    label_scores = model.get_outputs()[1].name

    print('[INFOR]: Load model success.')
    print('[INFOR]: Streaming...')
    # get output frame size
    _width, _height = args.input_shape

    input_path = str2int(args.input_data)
    # cap: globel variance
    cap = cv2.VideoCapture(input_path)

    print('[INFOR]: Start video capture.')
    Thread(target=video_capture, 
        args=(frame_queue, akadet_image_queue)).start()
    Thread(target=inference, 
        args=(akadet_image_queue, detections_queue, fps_queue)).start()
    Thread(target=drawing, 
        args=(frame_queue, detections_queue, fps_queue)).start()
    # end