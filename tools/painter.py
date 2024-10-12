# paint masks, contours, or points on images, with specified colors
import cv2
import numpy as np
from PIL import Image


def vis_add_mask_cv(image, mask, color: int, alpha):
	selected_color = cv2.applyColorMap((np.ones((1, 1)) * color).astype(np.uint8), cv2.COLORMAP_COOL)
	mask = mask > 0.5
	image[mask] = image[mask] * (1-alpha) + selected_color[0][0] * alpha
	return image.astype('uint8')

def point_painter(input_image, input_points, point_color=5, point_alpha=0.9, point_radius=15, contour_color=2, contour_width=5):
	h, w = input_image.shape[:2]
	point_mask = np.zeros((h, w)).astype('uint8')
	for point in input_points:
		point_mask[point[1], point[0]] = 1

	kernel = cv2.getStructuringElement(2, (point_radius, point_radius))
	point_mask = cv2.dilate(point_mask, kernel)

	contour_radius = (contour_width - 1) // 2
	dist_transform_fore = cv2.distanceTransform(point_mask, cv2.DIST_L2, 3)
	dist_transform_back = cv2.distanceTransform(1-point_mask, cv2.DIST_L2, 3)
	dist_map = dist_transform_fore - dist_transform_back
	# ...:::!!!:::...
	contour_radius += 2
	contour_mask = np.abs(np.clip(dist_map, -contour_radius, contour_radius))
	contour_mask = contour_mask / np.max(contour_mask)
	contour_mask[contour_mask>0.5] = 1.

	# paint mask
	painted_image = vis_add_mask_cv(input_image.copy(), point_mask, point_color, point_alpha)
	# paint contour
	painted_image = vis_add_mask_cv(painted_image.copy(), 1-contour_mask, contour_color, 1)
	return painted_image

def mask_painter_cv(input_image, input_mask, mask_color=5, mask_alpha=0.7, contour_color=1, contour_width=3):
	assert input_image.shape[:2] == input_mask.shape, 'different shape between image and mask'
	# 0: background, 1: foreground
	mask = np.clip(input_mask, 0, 1)
	contour_radius = (contour_width - 1) // 2
	
	dist_transform_fore = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
	dist_transform_back = cv2.distanceTransform(1-mask, cv2.DIST_L2, 3)
	dist_map = dist_transform_fore - dist_transform_back
	# ...:::!!!:::...
	contour_radius += 2
	contour_mask = np.abs(np.clip(dist_map, -contour_radius, contour_radius))
	contour_mask = contour_mask / np.max(contour_mask)
	contour_mask[contour_mask>0.5] = 1.

	# paint mask
	painted_image = vis_add_mask_cv(input_image.copy(), mask.copy(), mask_color, mask_alpha)
	# paint contour
	painted_image = vis_add_mask_cv(painted_image.copy(), 1-contour_mask, contour_color, 1)

	return painted_image

def background_remover(input_image, input_mask):
	"""
	input_image: H, W, 3, np.array
	input_mask: H, W, np.array

	image_wo_background: PIL.Image	
	"""
	assert input_image.shape[:2] == input_mask.shape, 'different shape between image and mask'
	# 0: background, 1: foreground
	mask = np.expand_dims(np.clip(input_mask, 0, 1), axis=2)*255
	image_wo_background = np.concatenate([input_image, mask], axis=2)		# H, W, 4
	image_wo_background = Image.fromarray(image_wo_background).convert('RGBA')

	return image_wo_background

