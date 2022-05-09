import cv2
import numpy as np
import pandas as pd


def visibility(image):
	height, width = image.shape[:2]
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	contrast_distribution = np.zeros((height, width), np.uint8)
	valid_area = cv2.imread('target2.png', 0)

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
					C_max = 255
				else:
					C_max = 0;

				contrast_distribution[r, c] = C_max

	contrast_distribution = 255 - contrast_distribution
	contrast_distribution = contrast_distribution & valid_area

	res = [0, 0]

	for r in range(1, np.size(contrast_distribution, 0)):
		poss_pt = np.where(contrast_distribution[r] > 0)
		if poss_pt:
			for i in range(len(poss_pt[0])):
				np_where = np.where(
					contrast_distribution[
					r:
					min(np.size(contrast_distribution, 0), r+50), max(1, poss_pt[0][i]-25):
					min(np.size(contrast_distribution, 1), poss_pt[1][i] + 25)] > 0)
				if np_where:
					np_where = np_where[0]
				else:
					np_where = 0
				valid_points_num = len(np_where)

				if valid_points_num > 3:
					t = np.where(contrast_distribution[r:r+10] > 0)
					if t:
						if len(t[0]) > 4:
							res = [r, poss_pt[i]]
							flag_break = True
							break

	return res, contrast_distribution


if __name__ == '__main__':
	image = cv2.imread('image2.jpg')
	height, width = image.shape[:2]
	print(width, height)
	res, contrast_distribution = visibility(image)
	print(res)
	cv2.line(contrast_distribution, (0, res[1]), (width, res[1]), (0, 0, 0), 5)
	cv2.imwrite('result2.jpg', contrast_distribution)
	cv2.imshow('Image', contrast_distribution)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
