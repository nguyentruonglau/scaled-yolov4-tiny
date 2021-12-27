from collections import OrderedDict
import copy
import time
import numpy as np

from pycocotools import coco
from pycocotools import cocoeval
from pycocotools import mask

import six
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf


class COCOWrapper(coco.COCO):
  """Wrapper for the pycocotools COCO class."""

  def __init__(self, dataset, detection_type='bbox'):
    """COCOWrapper constructor.

    See http://mscoco.org/dataset/#format for a description of the format.
    By default, the coco.COCO class constructor reads from a JSON file.
    This function duplicates the same behavior but loads from a dictionary,
    allowing us to perform evaluation without writing to external storage.

    Args:
      dataset: a dictionary holding bounding box annotations in the COCO format.
      detection_type: type of detections being wrapped. Can be one of ['bbox',
        'segmentation']

    Raises:
      ValueError: if detection_type is unsupported.
    """
    supported_detection_types = ['bbox', 'segmentation']
    if detection_type not in supported_detection_types:
      raise ValueError('Unsupported detection type: {}. '
                       'Supported values are: {}'.format(
                           detection_type, supported_detection_types))
    self._detection_type = detection_type
    coco.COCO.__init__(self)
    self.dataset = dataset
    self.createIndex()

  def LoadAnnotations(self, annotations):
    """Load annotations dictionary into COCO datastructure.

    See http://mscoco.org/dataset/#format for a description of the annotations
    format.  As above, this function replicates the default behavior of the API
    but does not require writing to external storage.

    Args:
      annotations: python list holding object detection results where each
        detection is encoded as a dict with required keys ['image_id',
        'category_id', 'score'] and one of ['bbox', 'segmentation'] based on
        `detection_type`.

    Returns:
      a coco.COCO datastructure holding object detection annotations results

    Raises:
      ValueError: if annotations is not a list
      ValueError: if annotations do not correspond to the images contained
        in self.
    """
    results = coco.COCO()
    results.dataset['images'] = [img for img in self.dataset['images']]

    tf.logging.info('Loading and preparing annotation results...')
    tic = time.time()

    if not isinstance(annotations, list):
      raise ValueError('annotations is not a list of objects')
    annotation_img_ids = [ann['image_id'] for ann in annotations]
    if (set(annotation_img_ids) != (set(annotation_img_ids)
                                    & set(self.getImgIds()))):
      raise ValueError('Results do not correspond to current coco set')
    results.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
    if self._detection_type == 'bbox':
      for idx, ann in enumerate(annotations):
        bb = ann['bbox']
        ann['area'] = bb[2] * bb[3]
        ann['id'] = idx + 1
        ann['iscrowd'] = 0
    elif self._detection_type == 'segmentation':
      for idx, ann in enumerate(annotations):
        ann['area'] = mask.area(ann['segmentation'])
        ann['bbox'] = mask.toBbox(ann['segmentation'])
        ann['id'] = idx + 1
        ann['iscrowd'] = 0
    tf.logging.info('DONE (t=%0.2fs)', (time.time() - tic))

    results.dataset['annotations'] = annotations
    results.createIndex()
    return results


class COCOEvalWrapper(cocoeval.COCOeval):
  """Wrapper for the pycocotools COCOeval class.

  To evaluate, create two objects (groundtruth_dict and detections_list)
  using the conventions listed at http://mscoco.org/dataset/#format.
  Then call evaluation as follows:

    groundtruth = coco_tools.COCOWrapper(groundtruth_dict)
    detections = groundtruth.LoadAnnotations(detections_list)
    evaluator = coco_tools.COCOEvalWrapper(groundtruth, detections,
                                           agnostic_mode=False)

    metrics = evaluator.ComputeMetrics()
  """

  def __init__(self, groundtruth=None, detections=None, agnostic_mode=False,
               iou_type='bbox', oks_sigmas=None):
    """COCOEvalWrapper constructor.

    Note that for the area-based metrics to be meaningful, detection and
    groundtruth boxes must be in image coordinates measured in pixels.

    Args:
      groundtruth: a coco.COCO (or coco_tools.COCOWrapper) object holding
        groundtruth annotations
      detections: a coco.COCO (or coco_tools.COCOWrapper) object holding
        detections
      agnostic_mode: boolean (default: False).  If True, evaluation ignores
        class labels, treating all detections as proposals.
      iou_type: IOU type to use for evaluation. Supports `bbox', `segm`,
        `keypoints`.
      oks_sigmas: Float numpy array holding the OKS variances for keypoints.
    """
    cocoeval.COCOeval.__init__(self, groundtruth, detections, iouType=iou_type)
    if oks_sigmas is not None:
      self.params.kpt_oks_sigmas = oks_sigmas
    if agnostic_mode:
      self.params.useCats = 0
    self._iou_type = iou_type

  def GetCategory(self, category_id):
    """Fetches dictionary holding category information given category id.

    Args:
      category_id: integer id
    Returns:
      dictionary holding 'id', 'name'.
    """
    return self.cocoGt.cats[category_id]

  def GetAgnosticMode(self):
    """Returns true if COCO Eval is configured to evaluate in agnostic mode."""
    return self.params.useCats == 0

  def GetCategoryIdList(self):
    """Returns list of valid category ids."""
    return self.params.catIds

  def ComputeMetrics(self,
                     include_metrics_per_category=False,
                     all_metrics_per_category=False,
                     super_categories=None):
    """Computes detection/keypoint metrics.

    Args:
      include_metrics_per_category: If True, will include metrics per category.
      all_metrics_per_category: If true, include all the summery metrics for
        each category in per_category_ap. Be careful with setting it to true if
        you have more than handful of categories, because it will pollute
        your mldash.
      super_categories: None or a python dict mapping super-category names
        (strings) to lists of categories (corresponding to category names
        in the label_map).  Metrics are aggregated along these super-categories
        and added to the `per_category_ap` and are associated with the name
          `PerformanceBySuperCategory/<super-category-name>`.

    Returns:
      1. summary_metrics: a dictionary holding:
        'Precision/mAP': mean average precision over classes averaged over IOU
          thresholds ranging from .5 to .95 with .05 increments
        'Precision/mAP@.50IOU': mean average precision at 50% IOU
        'Precision/mAP@.75IOU': mean average precision at 75% IOU
        'Precision/mAP (small)': mean average precision for small objects
                        (area < 32^2 pixels). NOTE: not present for 'keypoints'
        'Precision/mAP (medium)': mean average precision for medium sized
                        objects (32^2 pixels < area < 96^2 pixels)
        'Precision/mAP (large)': mean average precision for large objects
                        (96^2 pixels < area < 10000^2 pixels)
        'Recall/AR@1': average recall with 1 detection
        'Recall/AR@10': average recall with 10 detections
        'Recall/AR@100': average recall with 100 detections
        'Recall/AR@100 (small)': average recall for small objects with 100
          detections. NOTE: not present for 'keypoints'
        'Recall/AR@100 (medium)': average recall for medium objects with 100
          detections
        'Recall/AR@100 (large)': average recall for large objects with 100
          detections
      2. per_category_ap: a dictionary holding category specific results with
        keys of the form: 'Precision mAP ByCategory/category'
        (without the supercategory part if no supercategories exist).
        For backward compatibility 'PerformanceByCategory' is included in the
        output regardless of all_metrics_per_category.
        If evaluating class-agnostic mode, per_category_ap is an empty
        dictionary.
        If super_categories are provided, then this will additionally include
        metrics aggregated along the super_categories with keys of the form:
        `PerformanceBySuperCategory/<super-category-name>`

    Raises:
      ValueError: If category_stats does not exist.
    """
    self.evaluate()
    self.accumulate()
    self.summarize()

    summary_metrics = {}
    if self._iou_type in ['bbox', 'segm']:
      summary_metrics = OrderedDict([('Precision/mAP', self.stats[0]),
                                     ('Precision/mAP@.50IOU', self.stats[1]),
                                     ('Precision/mAP@.75IOU', self.stats[2]),
                                     ('Precision/mAP (small)', self.stats[3]),
                                     ('Precision/mAP (medium)', self.stats[4]),
                                     ('Precision/mAP (large)', self.stats[5]),
                                     ('Recall/AR@1', self.stats[6]),
                                     ('Recall/AR@10', self.stats[7]),
                                     ('Recall/AR@100', self.stats[8]),
                                     ('Recall/AR@100 (small)', self.stats[9]),
                                     ('Recall/AR@100 (medium)', self.stats[10]),
                                     ('Recall/AR@100 (large)', self.stats[11])])
    elif self._iou_type == 'keypoints':
      category_id = self.GetCategoryIdList()[0]
      category_name = self.GetCategory(category_id)['name']
      summary_metrics = OrderedDict([])
      summary_metrics['Precision/mAP ByCategory/{}'.format(
          category_name)] = self.stats[0]
      summary_metrics['Precision/mAP@.50IOU ByCategory/{}'.format(
          category_name)] = self.stats[1]
      summary_metrics['Precision/mAP@.75IOU ByCategory/{}'.format(
          category_name)] = self.stats[2]
      summary_metrics['Precision/mAP (medium) ByCategory/{}'.format(
          category_name)] = self.stats[3]
      summary_metrics['Precision/mAP (large) ByCategory/{}'.format(
          category_name)] = self.stats[4]
      summary_metrics['Recall/AR@1 ByCategory/{}'.format(
          category_name)] = self.stats[5]
      summary_metrics['Recall/AR@10 ByCategory/{}'.format(
          category_name)] = self.stats[6]
      summary_metrics['Recall/AR@100 ByCategory/{}'.format(
          category_name)] = self.stats[7]
      summary_metrics['Recall/AR@100 (medium) ByCategory/{}'.format(
          category_name)] = self.stats[8]
      summary_metrics['Recall/AR@100 (large) ByCategory/{}'.format(
          category_name)] = self.stats[9]
    if not include_metrics_per_category:
      return summary_metrics, {}
    if not hasattr(self, 'category_stats'):
      raise ValueError('Category stats do not exist')
    per_category_ap = OrderedDict([])
    super_category_ap = OrderedDict([])
    if self.GetAgnosticMode():
      return summary_metrics, per_category_ap
    for category_index, category_id in enumerate(self.GetCategoryIdList()):
      category = self.GetCategory(category_id)['name']
      # Kept for backward compatilbility
      per_category_ap['PerformanceByCategory/mAP/{}'.format(
          category)] = self.category_stats[0][category_index]
      if super_categories:
        for key in super_categories:
          if category in super_categories[key]:
            metric_name = 'PerformanceBySuperCategory/{}'.format(key)
            if metric_name not in super_category_ap:
              super_category_ap[metric_name] = 0
            super_category_ap[metric_name] += self.category_stats[0][
                category_index]
      if all_metrics_per_category:
        per_category_ap['Precision mAP ByCategory/{}'.format(
            category)] = self.category_stats[0][category_index]
        per_category_ap['Precision mAP@.50IOU ByCategory/{}'.format(
            category)] = self.category_stats[1][category_index]
        per_category_ap['Precision mAP@.75IOU ByCategory/{}'.format(
            category)] = self.category_stats[2][category_index]
        per_category_ap['Precision mAP (small) ByCategory/{}'.format(
            category)] = self.category_stats[3][category_index]
        per_category_ap['Precision mAP (medium) ByCategory/{}'.format(
            category)] = self.category_stats[4][category_index]
        per_category_ap['Precision mAP (large) ByCategory/{}'.format(
            category)] = self.category_stats[5][category_index]
        per_category_ap['Recall AR@1 ByCategory/{}'.format(
            category)] = self.category_stats[6][category_index]
        per_category_ap['Recall AR@10 ByCategory/{}'.format(
            category)] = self.category_stats[7][category_index]
        per_category_ap['Recall AR@100 ByCategory/{}'.format(
            category)] = self.category_stats[8][category_index]
        per_category_ap['Recall AR@100 (small) ByCategory/{}'.format(
            category)] = self.category_stats[9][category_index]
        per_category_ap['Recall AR@100 (medium) ByCategory/{}'.format(
            category)] = self.category_stats[10][category_index]
        per_category_ap['Recall AR@100 (large) ByCategory/{}'.format(
            category)] = self.category_stats[11][category_index]
    if super_categories:
      for key in super_categories:
        metric_name = 'PerformanceBySuperCategory/{}'.format(key)
        super_category_ap[metric_name] /= len(super_categories[key])
      per_category_ap.update(super_category_ap)
    return summary_metrics


def _ConvertBoxToCOCOFormat(box):
  """Converts a box in [ymin, xmin, ymax, xmax] format to COCO format.

  This is a utility function for converting from our internal
  [ymin, xmin, ymax, xmax] convention to the convention used by the COCO API
  i.e., [xmin, ymin, width, height].

  Args:
    box: a [ymin, xmin, ymax, xmax] numpy array

  Returns:
    a list of floats representing [xmin, ymin, width, height]
  """
  return [float(box[1]), float(box[0]), float(box[3] - box[1]),
          float(box[2] - box[0])]


def _RleCompress(masks):
  """Compresses mask using Run-length encoding provided by pycocotools.

  Args:
    masks: uint8 numpy array of shape [mask_height, mask_width] with values in
    {0, 1}.

  Returns:
    A pycocotools Run-length encoding of the mask.
  """
  rle = mask.encode(np.asfortranarray(masks))
  rle['counts'] = six.ensure_str(rle['counts'])
  return rle


def ExportSingleImageGroundtruthToCoco(image_id,
                                       next_annotation_id,
                                       category_id_set,
                                       groundtruth_boxes,
                                       groundtruth_classes,
                                       groundtruth_keypoints=None,
                                       groundtruth_keypoint_visibilities=None,
                                       groundtruth_masks=None,
                                       groundtruth_is_crowd=None,
                                       groundtruth_area=None):
  """Export groundtruth of a single image to COCO format.

  This function converts groundtruth detection annotations represented as numpy
  arrays to dictionaries that can be ingested by the COCO evaluation API. Note
  that the image_ids provided here must match the ones given to
  ExportSingleImageDetectionsToCoco. We assume that boxes and classes are in
  correspondence - that is: groundtruth_boxes[i, :], and
  groundtruth_classes[i] are associated with the same groundtruth annotation.

  In the exported result, "area" fields are always set to the area of the
  groundtruth bounding box.

  Args:
    image_id: a unique image identifier either of type integer or string.
    next_annotation_id: integer specifying the first id to use for the
      groundtruth annotations. All annotations are assigned a continuous integer
      id starting from this value.
    category_id_set: A set of valid class ids. Groundtruth with classes not in
      category_id_set are dropped.
    groundtruth_boxes: numpy array (float32) with shape [num_gt_boxes, 4]
    groundtruth_classes: numpy array (int) with shape [num_gt_boxes]
    groundtruth_keypoints: optional float numpy array of keypoints
      with shape [num_gt_boxes, num_keypoints, 2].
    groundtruth_keypoint_visibilities: optional integer numpy array of keypoint
      visibilities with shape [num_gt_boxes, num_keypoints]. Integer is treated
      as an enum with 0=not labels, 1=labeled but not visible and 2=labeled and
      visible.
    groundtruth_masks: optional uint8 numpy array of shape [num_detections,
      image_height, image_width] containing detection_masks.
    groundtruth_is_crowd: optional numpy array (int) with shape [num_gt_boxes]
      indicating whether groundtruth boxes are crowd.
    groundtruth_area: numpy array (float32) with shape [num_gt_boxes]. If
      provided, then the area values (in the original absolute coordinates) will
      be populated instead of calculated from bounding box coordinates.

  Returns:
    a list of groundtruth annotations for a single image in the COCO format.

  Raises:
    ValueError: if (1) groundtruth_boxes and groundtruth_classes do not have the
      right lengths or (2) if each of the elements inside these lists do not
      have the correct shapes or (3) if image_ids are not integers
  """

  if len(groundtruth_classes.shape) != 1:
    raise ValueError('groundtruth_classes is '
                     'expected to be of rank 1.')
  if len(groundtruth_boxes.shape) != 2:
    raise ValueError('groundtruth_boxes is expected to be of '
                     'rank 2.')
  if groundtruth_boxes.shape[1] != 4:
    raise ValueError('groundtruth_boxes should have '
                     'shape[1] == 4.')
  num_boxes = groundtruth_classes.shape[0]
  if num_boxes != groundtruth_boxes.shape[0]:
    raise ValueError('Corresponding entries in groundtruth_classes, '
                     'and groundtruth_boxes should have '
                     'compatible shapes (i.e., agree on the 0th dimension).'
                     'Classes shape: %d. Boxes shape: %d. Image ID: %s' % (
                         groundtruth_classes.shape[0],
                         groundtruth_boxes.shape[0], image_id))
  has_is_crowd = groundtruth_is_crowd is not None
  if has_is_crowd and len(groundtruth_is_crowd.shape) != 1:
    raise ValueError('groundtruth_is_crowd is expected to be of rank 1.')
  has_keypoints = groundtruth_keypoints is not None
  has_keypoint_visibilities = groundtruth_keypoint_visibilities is not None
  if has_keypoints and not has_keypoint_visibilities:
    groundtruth_keypoint_visibilities = np.full(
        (num_boxes, groundtruth_keypoints.shape[1]), 2)
  groundtruth_list = []
  for i in range(num_boxes):
    if groundtruth_classes[i] in category_id_set:
      iscrowd = groundtruth_is_crowd[i] if has_is_crowd else 0
      if groundtruth_area is not None and groundtruth_area[i] > 0:
        area = float(groundtruth_area[i])
      else:
        area = float((groundtruth_boxes[i, 2] - groundtruth_boxes[i, 0]) *
                     (groundtruth_boxes[i, 3] - groundtruth_boxes[i, 1]))
      export_dict = {
          'id':
              next_annotation_id + i,
          'image_id':
              image_id,
          'category_id':
              int(groundtruth_classes[i]),
          'bbox':
              list(_ConvertBoxToCOCOFormat(groundtruth_boxes[i, :])),
          'area': area,
          'iscrowd':
              iscrowd
      }
      if groundtruth_masks is not None:
        export_dict['segmentation'] = _RleCompress(groundtruth_masks[i])
      if has_keypoints:
        keypoints = groundtruth_keypoints[i]
        visibilities = np.reshape(groundtruth_keypoint_visibilities[i], [-1])
        coco_keypoints = []
        num_valid_keypoints = 0
        for keypoint, visibility in zip(keypoints, visibilities):
          # Convert from [y, x] to [x, y] as mandated by COCO.
          coco_keypoints.append(float(keypoint[1]))
          coco_keypoints.append(float(keypoint[0]))
          coco_keypoints.append(int(visibility))
          if int(visibility) > 0:
            num_valid_keypoints = num_valid_keypoints + 1
        export_dict['keypoints'] = coco_keypoints
        export_dict['num_keypoints'] = num_valid_keypoints

      groundtruth_list.append(export_dict)
  return groundtruth_list


def ExportGroundtruthToCOCO(image_ids,
                            groundtruth_boxes,
                            groundtruth_classes,
                            categories,
                            output_path=None):
  """Export groundtruth detection annotations in numpy arrays to COCO API.

  This function converts a set of groundtruth detection annotations represented
  as numpy arrays to dictionaries that can be ingested by the COCO API.
  Inputs to this function are three lists: image ids for each groundtruth image,
  groundtruth boxes for each image and groundtruth classes respectively.
  Note that the image_ids provided here must match the ones given to the
  ExportDetectionsToCOCO function in order for evaluation to work properly.
  We assume that for each image, boxes, scores and classes are in
  correspondence --- that is: image_id[i], groundtruth_boxes[i, :] and
  groundtruth_classes[i] are associated with the same groundtruth annotation.

  In the exported result, "area" fields are always set to the area of the
  groundtruth bounding box and "iscrowd" fields are always set to 0.
  TODO(jonathanhuang): pass in "iscrowd" array for evaluating on COCO dataset.

  Args:
    image_ids: a list of unique image identifier either of type integer or
      string.
    groundtruth_boxes: list of numpy arrays with shape [num_gt_boxes, 4]
      (note that num_gt_boxes can be different for each entry in the list)
    groundtruth_classes: list of numpy arrays (int) with shape [num_gt_boxes]
      (note that num_gt_boxes can be different for each entry in the list)
    categories: a list of dictionaries representing all possible categories.
        Each dict in this list has the following keys:
          'id': (required) an integer id uniquely identifying this category
          'name': (required) string representing category name
            e.g., 'cat', 'dog', 'pizza'
          'supercategory': (optional) string representing the supercategory
            e.g., 'animal', 'vehicle', 'food', etc
    output_path: (optional) path for exporting result to JSON
  Returns:
    dictionary that can be read by COCO API
  Raises:
    ValueError: if (1) groundtruth_boxes and groundtruth_classes do not have the
      right lengths or (2) if each of the elements inside these lists do not
      have the correct shapes or (3) if image_ids are not integers
  """
  category_id_set = set([cat['id'] for cat in categories])
  groundtruth_export_list = []
  image_export_list = []
  if not len(image_ids) == len(groundtruth_boxes) == len(groundtruth_classes):
    raise ValueError('Input lists must have the same length')

  # For reasons internal to the COCO API, it is important that annotation ids
  # are not equal to zero; we thus start counting from 1.
  annotation_id = 1
  for image_id, boxes, classes in zip(image_ids, groundtruth_boxes,
                                      groundtruth_classes):
    image_export_list.append({'id': image_id})
    groundtruth_export_list.extend(ExportSingleImageGroundtruthToCoco(
        image_id,
        annotation_id,
        category_id_set,
        boxes,
        classes))
    num_boxes = classes.shape[0]
    annotation_id += num_boxes

  groundtruth_dict = {
      'annotations': groundtruth_export_list,
      'images': image_export_list,
      'categories': categories
  }

  return groundtruth_dict


def ExportDetectionsToCOCO(image_ids,
                           detection_boxes,
                           detection_scores,
                           detection_classes,
                           categories,
                           output_path=None):
  """Export detection annotations in numpy arrays to COCO API.

  This function converts a set of predicted detections represented
  as numpy arrays to dictionaries that can be ingested by the COCO API.
  Inputs to this function are lists, consisting of boxes, scores and
  classes, respectively, corresponding to each image for which detections
  have been produced.  Note that the image_ids provided here must
  match the ones given to the ExportGroundtruthToCOCO function in order
  for evaluation to work properly.

  We assume that for each image, boxes, scores and classes are in
  correspondence --- that is: detection_boxes[i, :], detection_scores[i] and
  detection_classes[i] are associated with the same detection.

  Args:
    image_ids: a list of unique image identifier either of type integer or
      string.
    detection_boxes: list of numpy arrays with shape [num_detection_boxes, 4]
    detection_scores: list of numpy arrays (float) with shape
      [num_detection_boxes]. Note that num_detection_boxes can be different
      for each entry in the list.
    detection_classes: list of numpy arrays (int) with shape
      [num_detection_boxes]. Note that num_detection_boxes can be different
      for each entry in the list.
    categories: a list of dictionaries representing all possible categories.
      Each dict in this list must have an integer 'id' key uniquely identifying
      this category.
    output_path: (optional) path for exporting result to JSON

  Returns:
    list of dictionaries that can be read by COCO API, where each entry
    corresponds to a single detection and has keys from:
    ['image_id', 'category_id', 'bbox', 'score'].
  Raises:
    ValueError: if (1) detection_boxes and detection_classes do not have the
      right lengths or (2) if each of the elements inside these lists do not
      have the correct shapes or (3) if image_ids are not integers.
  """
  category_id_set = set([cat['id'] for cat in categories])
  detections_export_list = []
  if not (len(image_ids) == len(detection_boxes) == len(detection_scores) ==
          len(detection_classes)):
    raise ValueError('Input lists must have the same length')
  for image_id, boxes, scores, classes in zip(image_ids, detection_boxes,
                                              detection_scores,
                                              detection_classes):
    detections_export_list.extend(ExportSingleImageDetectionBoxesToCoco(
        image_id,
        category_id_set,
        boxes,
        scores,
        classes))

  return detections_export_list
#Copyright 2017 The TensorFlow Authors. All Rights Reserved.