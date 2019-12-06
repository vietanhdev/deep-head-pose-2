
import random
import numpy as np
import cv2
import imgaug as ia
from imgaug import augmenters as iaa

seq = [None]

def load_aug():

	sometimes = lambda aug: iaa.Sometimes(0.5, aug)

	seq[0] = iaa.Sequential(
		[
			# execute 0 to 5 of the following (less important) augmenters per image
			# don't execute all of them, as that would often be way too strong
			iaa.SomeOf((0, 5),
				[
					iaa.OneOf([
						iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 2.0
						iaa.AverageBlur(k=(2, 3)), # blur image using local means
						iaa.MedianBlur(k=(3, 5)), # blur image using local medians
					]),
					iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
					iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
					# search either for all edges or for directed edges,
					iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
					iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
					iaa.AddToHueAndSaturation((-10, 10)), # change hue and saturation
					# either change the brightness of the whole image (sometimes
					# per channel) or change the brightness of subareas
					iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
					iaa.Grayscale(alpha=(0.0, 0.5)),
				],
				random_order=True
			)
		],
		random_order=True
	)


def augment_img(img):
	if seq[0] is None:
		load_aug()
	aug_det = seq[0].to_deterministic() 
	image_aug = aug_det.augment_image( img )
	return image_aug


