"""Edge detection and deforming card images"""

import numpy as np
import cv2
import pickle
from pathlib import Path
import logging

# from draw import show_color

HERE = Path(__file__).parent
BASE_CONTOUR_PATH = HERE / "resources/base_contour.pickle"

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

with open(str(BASE_CONTOUR_PATH), mode="rb") as f:
    base_contour = pickle.load(f)


def apply_contrast(img, alpha=2, beta=-100):
    _ = img.astype("int16")
    # return np.clip(cv2.convertScaleAbs(_, alpha=alpha, beta=beta), 0, 255).astype(np.uint8)
    return np.clip((_ * alpha) + beta, 0, 255).astype(np.uint8)


def apply_blur(img, shift: int = 7):
    img = cv2.convertScaleAbs(img)
    return cv2.GaussianBlur(img, (shift, shift), 0)


def mono_grad(image, shift):
    kernel = np.ones((shift, shift), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)


def resize_with_fill(image: np.ndarray, target_width: int, target_height: int):
    original_height, original_width = image.shape[:2]

    # Calculate the ratio to scale the image
    ratio = min(target_width / original_width, target_height / original_height)

    # Calculate new size and resize the image
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    resized_image = cv2.resize(image, (new_width, new_height))

    # Calculate padding to center the image
    top = (target_height - new_height) // 2
    bottom = target_height - new_height - top
    left = (target_width - new_width) // 2
    right = target_width - new_width - left

    # Pad the resized image to the desired size with black color
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded_image


def flatten_color(img: np.ndarray) -> np.ndarray:
    return np.array([img[:, :, 0].flatten(), img[:, :, 1].flatten(), img[:, :, 2].flatten()]).T


# def compress_img(img: np.ndarray, cluster: int = 8) -> np.ndarray:
#     img_width = img.shape[0]
#     img_height = img.shape[1]
#     img_flatten = flatten_color(img)
#     kmeans = KMeans(8)
#     kmeans.fit(img_flatten)
#     img_copressed = kmeans.cluster_centers_.astype(int)[kmeans.labels_]
#     return img_copressed.reshape((img_width, img_height, 3)).astype(np.uint8)


def remove_short_contours(contours, threshold: int = 300) -> list[np.ndarray]:
    results = []
    for contour in contours:
        length = cv2.arcLength(contour, True)
        if length > threshold:
            results.append(contour)

    return results


def smooth_contours(contours, epsilon: int = 5) -> list[np.ndarray]:
    new_contours = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, epsilon, True)
        new_contours.append(approx)

    return new_contours


def find_rectangle_contours(contours: tuple[np.ndarray], target_contour: np.ndarray, threshold=0.1):
    matches = []
    for contour in contours:
        length = cv2.arcLength(contour, True)
        match_power = cv2.matchShapes(contour, target_contour, cv2.CONTOURS_MATCH_I1, 0)
        if match_power <= threshold:
            matches.append([contour, match_power, length])

    return sorted(matches, key=lambda x: x[1])


def angle_between(pt0, pt1, pt2) -> float:
    B = pt0[0]
    C = pt1[0]
    F = pt2[0]
    FC = F - C
    BC = B - C
    FC_norm = FC / np.linalg.norm(FC)
    BC_norm = BC / np.linalg.norm(BC)
    dot_product = np.dot(FC_norm, BC_norm)
    beta = np.arccos(dot_product)
    return np.degrees(beta)


def remove_flat_points(contour, threshold: float = 3) -> np.ndarray:
    pt_count = len(contour)
    results = []

    for i in range(2, pt_count + 1):
        pts0 = contour[i - 2]
        pts1 = contour[i - 1]
        pts2 = contour[i % pt_count]
        angle = angle_between(pts0, pts1, pts2)
        if np.abs(180 - angle) >= threshold:
            results.append(contour[i - 1])

    return np.array(results)


def get_coeffs(pt0, pt1):
    points = [pt0, pt1]
    x_coords, y_coords = zip(*points)
    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
    s, i = np.linalg.lstsq(A, y_coords, rcond=None)[0]
    a = s
    b = -1.0
    c = -i
    return a, b, c


def reset_orientation(contour):
    def is_clockwise(contour):
        return np.cross(contour[1] - contour[0], contour[-1] - contour[0]) > 0

    contour = np.squeeze(contour, axis=1)

    if not is_clockwise(contour):
        contour = np.flip(contour, axis=0)

    s = contour.sum(axis=1)
    top_left_idx = np.argmin(s)
    contour = np.roll(contour, -top_left_idx, axis=0)

    return np.expand_dims(contour, axis=1)


def get_corners_from_contour(contour):
    pt_count = len(contour)
    ranks = []
    for i in range(0, pt_count):
        pt0 = contour[i][0]
        pt1 = contour[(i + 1) % pt_count][0]
        distance = np.linalg.norm(pt0 - pt1)
        ranks.append([[pt0, pt1], distance, i])

    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)[:4]
    ranks = sorted(ranks, key=lambda x: x[2], reverse=False)
    corners = []
    for i in range(4):
        edge0 = ranks[i][0]
        edge1 = ranks[(i + 1) % 4][0]

        a1, b1, c1 = get_coeffs(edge0[0], edge0[1])
        a2, b2, c2 = get_coeffs(edge1[0], edge1[1])

        A = np.array([[a1, b1], [a2, b2]])
        B = np.array([c1, c2])

        intersect_pt = np.linalg.solve(A, B)
        corners.append(intersect_pt)

    return np.expand_dims(np.array(corners), axis=1).astype(np.int32)


def deform_img_to_card(
    img: np.ndarray,
    contour,
    src_shape: tuple[int, int] = (512, 512),
    dst_shape: tuple[int, int] = (630, 880),
):
    """
    Deforms image based on given points represent corners

    Args:
        img (np.ndarray): Image array
        contour (_type_): ndarray of points, 3 dimentions
        src_shape (tuple[int, int], optional): Shape of pints. Defaults to (512, 512).
        dst_shape (tuple[int, int], optional): SHape of out image. Defaults to (630, 880).

    Returns:
        _type_: Deformed image
    """
    img_shape: tuple[int, int] = img.shape[:2]
    img_offset = (max(img_shape) - min(img_shape)) // 2
    offset = (img_offset, 0) if img_shape[0] > img_shape[1] else (img_offset, 0)
    scale_factor = max(img_shape) / max(src_shape)

    src_points = np.squeeze(contour, axis=1).astype("float32") * np.array([scale_factor, scale_factor], dtype="float32") - np.array(
        offset, dtype="float32"
    )
    dst_points = np.array(
        [
            [0, 0],
            [dst_shape[0] - 1, 0],
            [dst_shape[0] - 1, dst_shape[1] - 1],
            [0, dst_shape[1] - 1],
        ],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(img, M, dst_shape)


def deform_card(img_path: str, output_shape: tuple[int, int] = (660, 880)) -> np.ndarray | None:
    """
    From the given image path, tries to find

    Args:
        img_path (str): Path to the image to deform

    Returns:
        np.ndarray | None: Deformed image or None if it fails to find best match
    """
    IMG_SIZE = (512, 512)
    img = resize_with_fill(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), IMG_SIZE[0], IMG_SIZE[1])
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # blurred_rgb = cv2.GaussianBlur(img, (9, 9), 0)
    blurred_hsv = cv2.GaussianBlur(img_hsv, (9, 9), 0)

    alpha = 1.3
    beta = -20
    img_h = apply_contrast(blurred_hsv[:, :, 0], alpha=alpha, beta=beta)
    img_s = apply_contrast(blurred_hsv[:, :, 1], alpha=alpha, beta=beta)
    img_v = apply_contrast(blurred_hsv[:, :, 2], alpha=alpha, beta=beta)
    edge1 = cv2.Canny(img_h, 50, 100)
    edge2 = cv2.Canny(img_s, 50, 100)
    edge3 = cv2.Canny(img_v, 50, 100)
    # edge_all = cv2.max(cv2.max(edge1, edge2), edge3)

    # contours, hierarchy = cv2.findContours(edge_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours_all = []
    contours, hierarchy = cv2.findContours(edge1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours_all.extend(list(contours))
    contours, hierarchy = cv2.findContours(edge2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours_all.extend(list(contours))
    contours, hierarchy = cv2.findContours(edge3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours_all.extend(list(contours))

    cnt_smoothed = smooth_contours(contours_all, epsilon=1)
    cnt_smoothed_cleaned = remove_short_contours(cnt_smoothed)
    found_contours = find_rectangle_contours(cnt_smoothed_cleaned, base_contour)
    if not len(found_contours) > 0:
        logger.warning("No contour was found. Exiting.")
        return None

    best_fit_contour = found_contours[0][0]
    # best_fit_contour_convex = cv2.convexHull(best_fit_contour)
    best_fit_contour = remove_flat_points(best_fit_contour, threshold=2)
    best_fit_contour = get_corners_from_contour(best_fit_contour)
    best_fit_contour = reset_orientation(best_fit_contour)
    raw_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    return deform_img_to_card(raw_img, best_fit_contour, dst_shape=output_shape)


# if __name__ == "__main__":
#     img_path = "/home/yuri/github.com/AoesJP/project_pokereader/data/white_bg/IMG_1488.jpeg"
#     show_color(deform_card(img_path))
