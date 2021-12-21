import numpy as np
'''Config file for Augmentation Data, used during training

  	Format:
	   p_xxx: probability of occurrence of xxx augmentation
	   (a,b): a->min, b->max

	Note that:
		If don't use xxx, set max or min value
		ex: we don't use eraser, set p_eraser=0.0
'''
##yolo anchors
yolo_anchors = np.array([[(9, 8), (11, 10), (14, 14)],
                        [(17, 18), (27, 29), (48, 59)]])

##random noise with Gaussian distribution
p_noise = 0.7
var_limit = (10.0, 50.0) #variance range for noise
per_channel = False #noise will be sampled for each channel independently, otherwise

##color jitter, randomly changes the brightness, contrast, and saturation of an image
p_colorjitter = 0.7
brightness = 0.5
contrast = 0.7
saturation = 0.7
hue = 0.4 #[-0.5, 0.5]

##random blur image with Gaussian distribution
p_blur = 0.7
blur_limit = (3,3) # kernel size
sigma_limit = (0.0, 0.1) # standard deviation

##affine transformation: translation + rotation + scaling + shear
p_affine = 0.0
#1.0 denotes "no change" and 0.5 is zoomed out to 50 percent of the original size
scale = (0.8, 1.2)
#for both x and y-axis
translate_percent = (0.0, 0.0) 
#rotation in degrees, rotation happens around the center of the image
rotate = (-0, 0)
#same rotation
shear = (-0, 0) 

##vertical flip
p_vflip = 0.2

##horizontal flip
p_hflip = 0.8

##perspective
p_perspective = 1.0
deviation = [0.005, 0.01]
pad_val = np.random.randint(0,255)

##False->accept parts of the image being outside the image plane, True->opposite
fit_output = False

##color value
cval = np.random.randint(0,255)

##min_area, if the erea of the converted boxes is <3000, it will be ignored
min_area = 100

##min visibility, if the erea of the boxes is <35% of the original erea, the boxes will be ignored
min_visibility = 0.7