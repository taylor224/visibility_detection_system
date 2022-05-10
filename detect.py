import cv2
import csv
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
	with open('distance.csv', 'r') as csvfile:
		distance = list(csv.reader(csvfile))

	image = cv2.imread('image.jpg')
	preview_image = cv2.imread('image.jpg')
	height, width = image.shape[:2]
	print('original size :', width, 'x', height)

	cv2.putText(preview_image, 'Distance Check', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10, cv2.LINE_AA)
	for d in distance:
		cv2.putText(preview_image, d[1] + 'm', (width - 200, int(int(d[0]) * (height / 10000))-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
		cv2.line(preview_image, (0, int(int(d[0]) * (height / 10000))), (width, int(int(d[0]) * (height / 10000))), (0, 0, 255), 5)
	cv2.imshow('Image', preview_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	res, contrast_distribution = visibility('target.png', image)
	print('position :', res)

	final_distance = 0
	for i in range(len(distance)):
		index = len(distance) - i - 1
		current_point = distance[index]
		current_point_position = int(current_point[0])
		current_point_distance = int(current_point[1])
		res_height_percent = res[1] / height * 10000

		if int(current_point[0]) < res_height_percent:
			# No data to calculate point ratio
			if index == len(distance) - 1:
				final_distance = current_point_distance
			else:
				previous_point = distance[index+1]
				previous_point_position = int(previous_point[0])
				previous_point_distance = int(previous_point[1])
				
				position_ratio = (previous_point_position - res_height_percent) / (previous_point_position - current_point_position)
				final_distance = (current_point_distance - previous_point_distance) * position_ratio
			break

	print('Distance :', '%sm' %final_distance)
	contrast_distribution = cv2.cvtColor(contrast_distribution, cv2.COLOR_GRAY2RGB)
	cv2.putText(contrast_distribution, 'Distance %sm' % '{:,.2f}'.format(final_distance), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10, cv2.LINE_AA)
	cv2.putText(image, 'Distance %sm' % '{:,.2f}'.format(final_distance), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10, cv2.LINE_AA)
	cv2.line(contrast_distribution, (0, res[1]), (width, res[1]), (0, 0, 255), 5)
	cv2.line(image, (0, res[1]), (width, res[1]), (0, 0, 255), 5)
	cv2.imwrite('result.jpg', contrast_distribution)
	cv2.imwrite('image_line.jpg', image)
	cv2.imshow('Image', contrast_distribution)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
