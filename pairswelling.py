# import module
from PIL import Image, ImageChops
from aem import con
import cv2
import numpy as np
import os
import glob

# Remove shadow
def remove_shadow(image):
	open_cv_image = np.array(image)
	open_cv_image = open_cv_image[:, :, ::-1].copy()
	# open_cv_image = np.asfarray(image)
	# Remove shadow
	rgb_planes = cv2.split(open_cv_image)

	result_planes = []
	result_norm_planes = []
	for plane in rgb_planes:
		dilated_img = cv2.dilate(plane, np.ones((3,3), np.uint8))
		bg_img = cv2.medianBlur(dilated_img, 21)
		diff_img = 255 - cv2.absdiff(plane, bg_img)
		norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
		result_planes.append(diff_img)
		result_norm_planes.append(norm_img)

	result = cv2.merge(result_planes)
	result_norm = cv2.merge(result_norm_planes)

	return Image.fromarray(result_norm)

#----------------------------------------------------------------
path_1 = "t_start"
path_2 = "t_end"
path_3 = "t_diff"


#loop for each file in a folder
for file in os.listdir(path_1):
	#In macOS, '.DS_Store' is a hidden file. Check with 'ls -a'
	if file == (path_1 or path_2 or path_3) + '.DS_Store':
		continue

	filename_1 = os.fsdecode(file)
	filename_2 = filename_1.replace("t04","t12")

	#assign the images
	img1 = Image.open(path_1 + "/" + filename_1)
	img2 = Image.open(path_2 + "/" + filename_2)

	#remove shadow
	shadow_out_img1 = remove_shadow(img1)
	shadow_out_img2 = remove_shadow(img2)

	#finding difference
	diff = ImageChops.difference(shadow_out_img1, shadow_out_img2)
	diff.save(path_3+ "/" +filename_1.replace(".tif", "_diff.png"))