import os
import albumentations as al
import cv2
import skimage.metrics as mtr
import statistics as stat
import matplotlib.pyplot as plt
import numpy as np

from random import shuffle
from augmentation_util import crop_circle_from_img
from constants import COMBINED_IMG_PATH

CLAHE_CLIP_LIMITS = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
CLAHE_CHUNK_SIZES = [(4, 4), (6, 6), (8, 8), (10, 10), (12, 12), (16, 16)]
FILES = os.listdir(COMBINED_IMG_PATH)
shuffle(FILES)
FILES = FILES[0:150]


def plot_mat(mat, x_labels, y_labels, title) -> None:
	fig, ax = plt.subplots()
	im = ax.imshow(mat, interpolation="nearest", cmap="plasma")

	ax.set_xticks(np.arange(len(y_labels)), labels=list(map(lambda v: str(v), y_labels)))
	ax.set_yticks(np.arange(len(x_labels)), labels=list(map(lambda v: str(v[0]), x_labels)))

	for i in range(len(y_labels)):
		for j in range(len(x_labels)):
			text = ax.text(j, i, mat[i][j], ha="center", va="center", color="w")

	ax.set_title(title)
	fig.tight_layout()
	plt.savefig(f"{title.replace(" ", "-")}.png")


ssim_red_mean_mat = []
ssim_green_mean_mat = []
ssim_blue_mean_mat = []
psnr_all_mean_mat = []
psnr_red_mean_mat = []
psnr_green_mean_mat = []
psnr_blue_mean_mat = []
ssim_red_med_mat = []
ssim_green_med_mat = []
ssim_blue_med_mat = []
psnr_all_med_mat = []
psnr_red_med_mat = []
psnr_green_med_mat = []
psnr_blue_med_mat = []

for cl in CLAHE_CLIP_LIMITS:
	cl_ssim_red_means = []
	cl_ssim_green_means = []
	cl_ssim_blue_means = []
	cl_psnr_all_means = []
	cl_psnr_red_means = []
	cl_psnr_green_means = []
	cl_psnr_blue_means = []
	cl_ssim_red_meds = []
	cl_ssim_green_meds = []
	cl_ssim_blue_meds = []
	cl_psnr_all_meds = []
	cl_psnr_red_meds = []
	cl_psnr_green_meds = []
	cl_psnr_blue_meds = []
	for chunk_size in CLAHE_CHUNK_SIZES:
		ssim_reds = []
		ssim_greens = []
		ssim_blues = []
		psnr_alls = []
		psnr_reds = []
		psnr_greens = []
		psnr_blues = []
		for file in FILES:
			img = cv2.imread(f"{COMBINED_IMG_PATH}/{file}", cv2.COLOR_BGR2RGB)
			img = crop_circle_from_img(img)
			transform = al.CLAHE(clip_limit=cl, tile_grid_size=chunk_size, p=1.0)
			img_out = transform(image=img)["image"]
			img_out = crop_circle_from_img(img_out)
			ssim_red = mtr.structural_similarity(img[:, :, 0], img_out[:, :, 0])
			ssim_green = mtr.structural_similarity(img[:, :, 1], img_out[:, :, 1])
			ssim_blue = mtr.structural_similarity(img[:, :, 2], img_out[:, :, 2])
			psnr_all = mtr.peak_signal_noise_ratio(img, img_out)
			psnr_red = mtr.peak_signal_noise_ratio(img[:, :, 0], img_out[:, :, 0])
			psnr_green = mtr.peak_signal_noise_ratio(img[:, :, 1], img_out[:, :, 1])
			psnr_blue = mtr.peak_signal_noise_ratio(img[:, :, 2], img_out[:, :, 2])
			ssim_reds.append(ssim_red)
			ssim_greens.append(ssim_green)
			ssim_blues.append(ssim_blue)
			psnr_alls.append(psnr_all)
			psnr_reds.append(psnr_red)
			psnr_greens.append(psnr_green)
			psnr_blues.append(psnr_blue)
			print(file)
		cl_ssim_red_means.append(stat.mean(ssim_reds))
		cl_ssim_green_means.append(stat.mean(ssim_blues))
		cl_ssim_blue_means.append(stat.mean(ssim_greens))
		cl_psnr_all_means.append(stat.mean(psnr_alls))
		cl_psnr_red_means.append(stat.mean(psnr_reds))
		cl_psnr_green_means.append(stat.mean(psnr_greens))
		cl_psnr_blue_means.append(stat.mean(psnr_blues))
		cl_ssim_red_meds.append(stat.median(ssim_reds))
		cl_ssim_green_meds.append(stat.median(ssim_greens))
		cl_ssim_blue_meds.append(stat.median(ssim_blues))
		cl_psnr_all_meds.append(stat.median(psnr_alls))
		cl_psnr_red_meds.append(stat.median(psnr_reds))
		cl_psnr_green_meds.append(stat.median(psnr_greens))
		cl_psnr_blue_meds.append(stat.median(psnr_blues))
		print(chunk_size)
	ssim_red_mean_mat.append(cl_ssim_red_means)
	ssim_green_mean_mat.append(cl_ssim_green_means)
	ssim_blue_mean_mat.append(cl_ssim_blue_means)
	psnr_all_mean_mat.append(cl_psnr_all_means)
	psnr_red_mean_mat.append(cl_psnr_red_means)
	psnr_green_mean_mat.append(cl_psnr_green_means)
	psnr_blue_mean_mat.append(cl_psnr_blue_means)
	ssim_red_med_mat.append(cl_ssim_red_meds)
	ssim_green_med_mat.append(cl_ssim_green_meds)
	ssim_blue_med_mat.append(cl_ssim_blue_meds)
	psnr_all_med_mat.append(cl_psnr_all_meds)
	psnr_red_med_mat.append(cl_psnr_red_meds)
	psnr_green_med_mat.append(cl_psnr_green_meds)
	psnr_blue_med_mat.append(cl_psnr_blue_meds)
	print(cl)


plot_mat(ssim_red_mean_mat, CLAHE_CHUNK_SIZES, CLAHE_CLIP_LIMITS, "ssim red channel mean")
plot_mat(ssim_green_mean_mat, CLAHE_CHUNK_SIZES, CLAHE_CLIP_LIMITS, "ssim green channel mean")
plot_mat(ssim_blue_mean_mat, CLAHE_CHUNK_SIZES, CLAHE_CLIP_LIMITS, "ssim blue channel mean")
plot_mat(psnr_all_mean_mat, CLAHE_CHUNK_SIZES, CLAHE_CLIP_LIMITS, "psnr all mean")
plot_mat(psnr_red_mean_mat, CLAHE_CHUNK_SIZES, CLAHE_CLIP_LIMITS, "psnr red channel mean")
plot_mat(psnr_green_mean_mat, CLAHE_CHUNK_SIZES, CLAHE_CLIP_LIMITS, "psnr green channel mean")
plot_mat(psnr_blue_mean_mat, CLAHE_CHUNK_SIZES, CLAHE_CLIP_LIMITS, "psnr blue channel mean")
plot_mat(ssim_red_med_mat, CLAHE_CHUNK_SIZES, CLAHE_CLIP_LIMITS, "ssim red channel med")
plot_mat(ssim_green_med_mat, CLAHE_CHUNK_SIZES, CLAHE_CLIP_LIMITS, "ssim green channel med")
plot_mat(ssim_blue_med_mat, CLAHE_CHUNK_SIZES, CLAHE_CLIP_LIMITS, "ssim blue channel med")
plot_mat(psnr_all_med_mat, CLAHE_CHUNK_SIZES, CLAHE_CLIP_LIMITS, "psnr all med")
plot_mat(psnr_red_med_mat, CLAHE_CHUNK_SIZES, CLAHE_CLIP_LIMITS, "psnr red channel med")
plot_mat(psnr_green_med_mat, CLAHE_CHUNK_SIZES, CLAHE_CLIP_LIMITS, "psnr green channel med")
plot_mat(psnr_blue_med_mat, CLAHE_CHUNK_SIZES, CLAHE_CLIP_LIMITS, "psnr blue channel med")
