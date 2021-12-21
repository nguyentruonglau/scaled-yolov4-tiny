from generator import config_data as cfg
import albumentations as alb
from PIL import Image

import numpy as np
import random
import cv2
import copy


def alb_augmentation(image, bboxes, class_labels):
	"""Augmentation data with albumentations
	"""
	transform = alb.Compose(
	[
		#compose: affine, perspective, flip, noise, blur, color jitter
		alb.augmentations.geometric.transforms.Affine(
			scale=cfg.scale, 
			translate_percent=cfg.translate_percent,
			rotate=cfg.rotate,
			shear=cfg.shear,
			fit_output=cfg.fit_output,
			cval=cfg.cval,
			p=cfg.p_affine
		),
		alb.augmentations.geometric.transforms.Perspective(
            scale=cfg.deviation, 
            p=cfg.p_perspective, 
            fit_output=cfg.fit_output),
	  	alb.augmentations.transforms.HorizontalFlip(p=cfg.p_hflip),
	  	alb.augmentations.transforms.VerticalFlip(p=cfg.p_vflip),

	  	alb.augmentations.transforms.GaussNoise(var_limit=cfg.var_limit, 
	    	per_channel=cfg.per_channel, 
	    	p=cfg.p_noise),
	  	alb.augmentations.transforms.GaussianBlur(p=cfg.p_blur, 
            blur_limit=cfg.blur_limit, 
            sigma_limit=cfg.sigma_limit),
	  	alb.augmentations.transforms.ColorJitter(brightness=cfg.brightness, 
	    	contrast=cfg.contrast, 
	    	saturation=cfg.saturation, 
	    	hue=cfg.hue, 
	    	p=cfg.p_colorjitter)
	],
	#erea of boxes <min_erea and <min_visibility will be ignored
	bbox_params=alb.BboxParams(
		format='pascal_voc', 
		min_area=cfg.min_area, min_visibility=cfg.min_visibility, 
		label_fields=['class_labels'])
	)
	#apply transform
	transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
	#get data from transformed
	image = transformed['image']
	bboxes = transformed['bboxes']
	class_labels = transformed['class_labels']
	return image, bboxes, class_labels