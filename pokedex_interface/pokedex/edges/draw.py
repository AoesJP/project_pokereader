"""Helper module to draw images"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def show_color(img, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(img, vmin=0, vmax=255)


def show_grey(img, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap="grey", vmin=0, vmax=255)


def show_channels(img: np.ndarray, figsize=(10, 10)):
    plt.figure(figsize=figsize)
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


def draw_contours(img, contours, points: bool = True, line_width: int = 2, point_size: int = 6):
    img_copy = img.copy()
    img_contour = cv2.drawContours(img_copy, contours, -1, (50, 255, 0), line_width)
    for contour in contours:
        if points:
            for point in contour:
                cv2.circle(img_contour, tuple(point[0]), point_size, (255, 0, 0), -1)
    plt.imshow(img_contour)


def draw_lines(img: np.ndarray, lines: np.ndarray):
    img_copy = img.copy()
    for line in lines:
        a, b, c = line
        x0 = -a * c
        y0 = -b * c
        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * (a))
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * (a))
        cv2.line(img_copy, (x1, y1), (x2, y2), (255, 0, 255), 2)
    plt.imshow(img_copy)
