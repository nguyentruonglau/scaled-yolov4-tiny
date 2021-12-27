# SCALED YOLOV4 TINY

SIMPLER, FASTER AND MORE ACCURATE

[![Python 3.7](https://img.shields.io/badge/Python-3.7-3776AB)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow 2.4](https://img.shields.io/badge/TensorFlow-2.4-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Coverage](https://github.com/nguyentruonglau/scaled-yolov4-tiny/blob/main/images/coverage-93%25-green.svg)](https://github.com/nguyentruonglau/scaled-yolov4-tiny)
[![License](https://github.com/nguyentruonglau/scaled-yolov4-tiny/blob/main/images/license-MIT-green.svg)](https://github.com/nguyentruonglau/scaled-yolov4-tiny/blob/main/LICENSE.txt)

## 1. Model

Scaled YoloV4 Tiny, this is the implementation of "[Scaled-YOLOv4: Scaling Cross Stage Partial Network](https://arxiv.org/abs/2011.08036)" using Tensorflow framework.

| Model | Test Size | AP<sup>test</sup> | AP<sup>test</sup>(TTA) | FPS | GPU | CPU | PARAM | CAPACITY |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|  |  |  |  |  |  |  |
| **[Scaled YoloV4 Tiny](https://drive.google.com/file/d/1j8BKl18zl60q6dQLwegKK2aYi_-znLrX/view?usp=sharing)** | 416 | **21.7%** | **23%** | 100 *fps* (GPU) & 60 *fps* (CPU) | 1 Geforce RTX 3090Ti | Intel Core i7 10700K | 5.8M  | 23.1MB |

The recorded speed is based on the process we tested on the computer with CPU & GPU information as shown in the table above (with onnx & onnxruntime, FP32 and FPS = 1/(inference time + diou nms time), max can reach 252 FPS, without tensorrt)). Also acording to the "[Scaled-YOLOv4: Scaling Cross Stage Partial Network](https://arxiv.org/abs/2011.08036)" paper,  AP<sup>test</sup> of [YoloV3 Tiny](https://arxiv.org/abs/1804.02767) achieved 16.6%.

**Note that: pretrained models are trained on COCO dataset.**

## 2. Installation

```
   # Step 1: Cloning the repo
   
      >https://github.com/qai-research/scaled-yolov4-tiny

   # Step 2: Installing packages
  
      >python -m venv <virtual environments name>
      >activate.bat [in scripts folder]
      >pip install -r require.txt
      
   # Step 3: Installing Albumentations
      >pip install -U git+https://github.com/albumentations-team/albumentations
```

## 3. Checking enviroment

```
   # Step 1: Check if Tensorflow detected the GPU or not, make sure that you have (Cuda & Cuda Driver, CuDNN):
   
      >import tensorflow as tf
      >tf.config.list_physical_devices('GPU')
   
   # Step 2: If it returns: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
   then one GPU available.
```

## 4. Directory structure
```
├── database
    ├── dataset ── <datasetname> ├── Annotations (contain XML files)
    │                            ├── ClassNames (contain class names of dataset)
    │                            ├── ImageSets (train.txt, val.txt -> utilitys/split_dataset.py )
    │                            ├── JPEGImages (contain JPG images)
    │
    ├── images ├── input (contain images for detect.py)
    │          └── output (output of detect.py)
    │
    ├── models ├── checkpoint (save checkpoint during training)
    |          ├── pretrained (on Coco dataset)
    │          ├── best (save the best model)
    │          └── onnx (tf2onnx)
    │
    ├── logdata (contain data of explore_data.py)
    │
    └── videos ├── input (contain videos for video.py)
               └── output (output of video.py)
```

## 5. Usage

### Prepare Data
Scaled YoloV4 Tiny requires JPG format for images, Pascal VOC (XML) format for annotaions. If your image data is PNG for images or TXT|JSON for annotations, we provide tools to help you convert to. With images at [here](https://github.com/nguyentruonglau/png2jpg) and with annotations for [json Coco format](https://github.com/nguyentruonglau/json2xml) and [txt Coco format](https://github.com/nguyentruonglau/txt2xml). Also we have some useful tools: auto split train test, explore data,... at **utility** directory.


### Training

```
   # Step 1: Downloading pretrained backbone on COCO dataset, provided at the first table above.
   
   # Step 2: Tuning model hyperparameters
   
   # Step 3: Training
      > python train.py --model-shape (model shape must: %32 == 0)
                        --pretrained-weights (path to pretrained model on CoCo or checkpoint model)
                        --skip-difficult (if True, model will ignore objects that are difficult to detect)
                        --tensorboard (real time model evaluation, see more: tensorboard --logdir logtrain)
                        ...
```

### Inference For Images
```
   # Step 1: Preparing images in database/images/input
   # Step 2: Inference
      > python detect.py --pic-dir (contain jpg images)
                         --model-path (path to best model of you)
                         --nms-iou-threshold (greater than threshold, consider non-intersecting)
                         --nms-score-threshold (smaller than thresold, not use)
  # Step 3: See result at database/images/output
```

### Inference For Video

```
   # Step 1: Preparing videos in database/videos/input
   # Step 2: Stream camera or video
      > python video.py  --dont-show (if true, we don't see video streamed)
                         --input-shape (width and height of video, right click on video -> properties)
                         --nms (diou_nms will faster than hard_nms)
   # Step 3: See result at database/videos/output
```


<!-- LICENSE -->
## License

Distributed under the MIT License. See [LICENSE](https://github.com/qai-research/scaled-yolov4-tiny/blob/main/LICENSE.txt) for more information. Thanks Mr.Hieu (Thai Trung Hieu) contributing to this repository.


## Acknowledgements

[1]. Scaled-YOLOv4: Scaling Cross Stage Partial Network [22/02/2021] [paper](https://arxiv.org/abs/2011.08036)

[2]. YOLOv4: Optimal Speed and Accuracy of Object Detection [23/04/2020] [paper](https://arxiv.org/abs/2004.10934)

[3]. EfficientDet: Scalable and Efficient Object Detection [27/07/2020] [paper](https://arxiv.org/abs/1911.09070)

## Errors:

```
1. "ResourceExhaustedError: OOM when allocating tensor with shape[16,1024,13,13] and type float on 
/job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc [Op:Conv2D]"
-> GPU not enough space [reduce batchsize or reduce image size] (train.py).

2. "Tensorflow.python.framework.errors_impl.InvalidArgumentError: ConcatOp : Dimensions of inputs 
should match: shape[0] = [9,13,13,512] vs. shape[1] = [9,12,12,512] [Op:ConcatV2] name: concat" 
-> Check augment: model-shape (input model shape): Input data shapes for training, 
use 320+32*i(i>=0).

3. "Dimensions of inputs should match: shape[0] = [9,13,13,512] vs. shape[1] = [9,12,12,512] 
not matched" -> Check pretrained model with model type (train.py).

4. "ValueError: Could not find matching function to call loaded from the SavedModel" -> dtype of image
must is np.float32 -> img = np.array(img, dtype=np.float32) OR img.shape must have 4 dimensions 
[N, H, W, C] (train.py).

5. "Train_data[ob_name] += 1 KeyError: '折戸'"-> <>.names in ClassNamnes folder must from <>.txt, 
insert information to <>.names (split_dataset.py).

6. "Anchor_generator grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]][grid_anchor_index]
[5+batch_boxes[batch_index][box_index][4].astype(np.int32)] = 1
IndexError: index 9 is out of bounds for axis 0 with size 9" -> check num_classes argument
```

## Example

![](https://github.com/nguyentruonglau/scaled-yolov4-tiny/blob/main/images/demo.png)
