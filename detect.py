import cv2
import numpy as np
import pandas as pd
from itertools import groupby
from operator import itemgetter


DARK_RATIO_THRESHOLD = 0.05
# Adjust sensitivity of distance detection
# Lower will be more sensitive
WHITE_RATIO_THRESHOLD = 0.35


def visibility(target, image):
	height, width = image.shape[:2]
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	contrast_distribution = np.zeros((height, width), np.uint8)
	valid_area = cv2.imread(target, 0)

	for r in range(2, height-1):
		for c in range(2, width-1):
			points = np.array([np.array([r-1, c]),np.array([r, c-1]),np.array([r, c+1]),np.array([r+1, c])])
			C_max = -1;
			for idx in range(int(points.size / np.size(points, 1))):
				p = points[idx]
				divide_value = float(max(float(gray[r, c]), float(gray[p[0], p[1]])))

				if divide_value == 0:
					continue

				C_max = max(C_max, abs(float(gray[r, c]) - float(gray[p[0], p[1]])) / divide_value)

				if C_max > 0.05:
					C_max = C_max * 255
				else:
					C_max = 0;

				contrast_distribution[r, c] = C_max

	contrast_distribution = 255 - contrast_distribution
	contrast_distribution = contrast_distribution & valid_area

	start_res = [0, 0]
	res = [0, 0]

	#  To find first place of masking point
	for r in range(1, np.size(contrast_distribution, 0)):
		find_white = np.where(contrast_distribution[r] > 0)
		if find_white:
			find_white = find_white[0]

			if len(find_white) > 1:
				start_res = [find_white[0], r]
				break

	print('start res', start_res)

	for r in range(start_res[1], np.size(contrast_distribution, 0)):
		white = np.where(contrast_distribution[r] > 200)[0]
		dark = np.where((contrast_distribution[r] <= 200) & (contrast_distribution[r] != 0))[0]
		total_length = (len(dark) + len(white))
		# dark_white_ratio = len(dark) / (len(white) | 1)

		# if dark_white_ratio > DARK_RATIO_THRESHOLD:
		# 	continue

		is_not_valid = False
		for kk, gg in groupby(enumerate(dark), lambda ix : ix[0] - ix[1]):
			if len(list(map(itemgetter(1), gg))) > total_length / 70:
				is_not_valid = True
				break

		if not is_not_valid:
			for kk, gg in groupby(enumerate(white), lambda ix : ix[0] - ix[1]):
				if len(list(map(itemgetter(1), gg))) > total_length * WHITE_RATIO_THRESHOLD:
					is_not_valid = True
					break

		if not is_not_valid:
			res = [0, r]
			return res, contrast_distribution
	return res, contrast_distribution


if __name__ == '__main__':
	image = cv2.imread('image3.jpg')
	height, width = image.shape[:2]
	print(width, height)
	res, contrast_distribution = visibility('target3.png', image)
	print(res)
	contrast_distribution = cv2.cvtColor(contrast_distribution, cv2.COLOR_GRAY2RGB)
	cv2.line(contrast_distribution, (0, res[1]), (width, res[1]), (0, 0, 255), 5)
	cv2.line(image, (0, res[1]), (width, res[1]), (0, 0, 255), 5)
	cv2.imwrite('result3.jpg', contrast_distribution)
	cv2.imwrite('image_line3.jpg', image)
	cv2.imshow('Image', contrast_distribution)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
