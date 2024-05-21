"""Helper module to draw images"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def show_color(img):
    plt.imshow(img, vmin=0, vmax=255)


def show_grey(img):
    plt.imshow(img, cmap="grey", vmin=0, vmax=255)


def show_channels(img: np.ndarray):
    channels = img.shape[2]
    plt.figure(figsize=(6 * channels, 6))
    for i in range(channels):
        plt.subplot(int(f"1{channels}{i+1}"))
        plt.imshow(img[:, :, i], cmap="grey", vmin=0, vmax=255)


def draw_contour(img, contour, points: bool = True, line_width: int = 2, point_size: int = 6):
    img_copy = img.copy()
    img_contour = cv2.drawContours(img_copy, [contour], -1, (50, 255, 0), line_width)
    if points:
        for point in contour:
            cv2.circle(img_contour, tuple(point[0]), point_size, (255, 0, 0), -1)
    plt.imshow(img_contour)
