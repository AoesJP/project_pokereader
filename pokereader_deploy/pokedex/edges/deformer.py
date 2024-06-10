"""Edge detection and deforming card images"""

import numpy as np
import cv2
import pickle
from pathlib import Path
import logging
from PIL import Image, ImageOps
from pokedex import HIRES_HEIGHT, HIRES_WIDTH


# from draw import show_color

HERE = Path(__file__).parent
BASE_CONTOUR_PATH = HERE / "resources/base_contour.pickle"

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

with open(str(BASE_CONTOUR_PATH), mode="rb") as f:
    base_contour = pickle.load(f)


def apply_contrast(img: np.ndarray, alpha=2, beta=-100) -> np.ndarray:
    _ = img.astype("int16")
    # return np.clip(cv2.convertScaleAbs(_, alpha=alpha, beta=beta), 0, 255).astype(np.uint8)
    return np.clip((_ * alpha) + beta, 0, 255).astype(np.uint8)


def apply_blur(img: np.ndarray, shift: int = 7) -> np.ndarray:
    img = cv2.convertScaleAbs(img)
    return cv2.GaussianBlur(img, (shift, shift), 0)


def mono_grad(image: np.ndarray, shift) -> np.ndarray:
    """
    Extracting edges based on morphologyEx
    """
    kernel = np.ones((shift, shift), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)


def resize_with_fill(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Resizing image with padding
    """

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


def remove_short_long_contours(contours, resolution: tuple[int, int], min_length: int = 300, max_ratio: float = 0.1) -> list[np.ndarray]:
    """
    Removing contours based on length
    """
    results = []
    max_length = resolution[0] * 2 + resolution[1] * 2
    for contour in contours:
        length = cv2.arcLength(contour, True)
        if (length > min_length) and (length < max_length * (1 - max_ratio)):
            results.append(contour)

    return results


def smooth_contours(contours, ratio: float = 0.0005) -> list[np.ndarray]:
    """
    Smoothing contours
    """
    new_contours = []
    for contour in contours:
        arc_length = cv2.arcLength(contour, closed=True)
        epsilon = np.sqrt(arc_length) * ratio
        approx = cv2.approxPolyDP(contour, epsilon, True)
        new_contours.append(approx)

    return new_contours


def find_rectangle_contours(contours: tuple[np.ndarray], target_contour: np.ndarray, threshold=0.1) -> list[np.ndarray]:
    """
    Find countours which match the traget contour shape
    """
    matches = []
    for contour in contours:
        length = cv2.arcLength(contour, True)
        match_power = cv2.matchShapes(contour, target_contour, cv2.CONTOURS_MATCH_I1, 0)
        if match_power <= threshold:
            matches.append([contour, match_power, length])

    return sorted(matches, key=lambda x: x[1])


def angle_between(pt0, pt1, pt2) -> float:
    """
    Calculate angle between vectors (pt1, pt0) and (pt1, pt2)
    """
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
    """
    Remove points on a given contour based on angle
    """
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


def get_coeffs(pt0, pt1) -> tuple[float, float, float]:
    """
    Calculate slopes for x and y, and intercept
    """
    points = [pt0, pt1]
    x_coords, y_coords = zip(*points)
    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
    s, i = np.linalg.lstsq(A, y_coords, rcond=None)[0]
    a = s
    b = -1.0
    c = -i
    return a, b, c


def reset_orientation(contour) -> np.ndarray:
    """
    Reset contour orientation and make it clock wise orientation
    """

    def is_clockwise(contour):
        return np.cross(contour[1] - contour[0], contour[-1] - contour[0]) > 0

    contour = np.squeeze(contour, axis=1)

    if not is_clockwise(contour):
        contour = np.flip(contour, axis=0)

    s = contour.sum(axis=1)
    top_left_idx = np.argmin(s)
    contour = np.roll(contour, -top_left_idx, axis=0)

    return np.expand_dims(contour, axis=1)


def remove_protrude(img: np.ndarray, contour: np.ndarray, margin: int = 3) -> np.ndarray:
    """
    Remove protruding points
    """
    img_mask = np.zeros_like(img[:, :, 0])
    cv2.drawContours(img_mask, [contour], -1, (255), thickness=cv2.FILLED)
    kernel = np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)
    img_mask = cv2.erode(img_mask, kernel, iterations=1)
    img_mask = cv2.dilate(img_mask, kernel, iterations=2)
    img_shape = img_mask.shape

    ctr = contour.squeeze()
    ctr = ctr[(ctr[:, 0] >= 0) & (ctr[:, 0] <= img_shape[1]) & (ctr[:, 1] >= 0) & (ctr[:, 1] <= img_shape[0])]
    return np.expand_dims(ctr[img_mask[ctr[:, 1], ctr[:, 0]] > 0], axis=1)


def calc_lines(lines: np.ndarray) -> np.ndarray:
    """
    Calculating line slopes for x and y and intercept from rho and theta
    """
    lines = lines.squeeze()
    new_lines = []
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        c = -rho
        new_lines.append([a, b, c])

    return np.array(new_lines, dtype="float32")


def find_intersection(a: tuple, b: tuple) -> np.ndarray:
    """
    Find intersections of given two vectors
    """
    a1, b1, c1 = a[0], a[1], a[2]
    a2, b2, c2 = b[0], b[1], b[2]
    A = np.array([[a1, b1], [a2, b2]])
    B = np.array([-c1, -c2])

    try:
        intersection_point = np.linalg.solve(A, B)
        return intersection_point
    except np.linalg.LinAlgError:
        raise ValueError("The lines are parallel and do not intersect.")


def expand_edges(img: np.ndarray, kernel=(2, 2), iterations=1) -> np.ndarray:
    """
    Expand edges
    """
    return cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_RECT, ksize=kernel), iterations=iterations)


def get_corners_from_contour(contour) -> np.ndarray:
    """
    Calculate corner points from given contour
    Expecting the contour to have only 4 points
    """
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


def get_corners_from_lines(lines: list[np.ndarray], contour: np.ndarray) -> np.ndarray:
    """
    Calculate corner points from given lines
    """

    def line_equation(slope, point):
        m = slope
        x1, y1 = point
        a = m
        b = -1
        c = y1 - m * x1

        return a, b, c

    # def average_line(lines: np.ndarray):
    #     bs = lines[:, 1]
    #     count = bs.shape[0]
    #     bs_array = np.tile(bs, count).reshape((count, count))
    #     ident_mat = np.identity(count)
    #     flip_mat = 1 - ident_mat
    #     bs_array *= flip_mat
    #     bs_array += ident_mat
    #     factor = bs_array.prod(axis=1).reshape((count, 1))
    #     factor = factor / np.linalg.norm(factor)
    #     result = np.dot(lines.T, factor).squeeze()
    #     result_norm = np.linalg.norm(result[:2])
    #     return result / result_norm

    def average_line(lines: np.ndarray):
        c_sign = (2 * (lines[:, 2] >= 0).astype("int8") - 1).reshape((len(lines), 1))
        lines_mod = lines * c_sign
        a_mean = np.mean(lines_mod[:, 0])
        b_mean = np.mean(lines_mod[:, 1])
        c_mean = np.mean(lines_mod[:, 2])
        return np.array([a_mean, b_mean, c_mean])

    lines_r = []
    lines_l = []
    lines_t = []
    lines_b = []

    vx, vy, x, y = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)

    # main_inter = main_b[0] / main_a[0]
    # sub_inter = -1 / main_inter

    main_d = np.array((vy[0], vx[0])) / np.linalg.norm((vy[0], vx[0]), ord=2)
    sub_d = np.array((vx[0], -vy[0])) / np.linalg.norm((vx[0], -vy[0]), ord=2)
    main_a, main_b, main_c = line_equation(vy[0] / vx[0], (x[0], y[0]))
    sub_a, sub_b, sub_c = line_equation(sub_d[0] / sub_d[1], (x[0], y[0]))
    # print(main_a, main_b, main_c)
    # print(sub_a, sub_b, sub_c)

    center = contour.squeeze().mean(axis=0)

    for line in lines:
        a, b, c = line
        # lines_r.append(val_x)

        line_d = np.array((a, -b)) / np.linalg.norm((a, -b))
        # val_x = a * center[0] + b * center[1] + c

        dot_prd = np.dot(line_d, main_d)
        if abs(dot_prd) > 0.5:  # Vertical Lines
            intersect = find_intersection((sub_a, sub_b, sub_c), (a, b, c))
            if intersect[0] < center[0]:  # Left
                # print("Left")
                lines_l.append(line)
            else:  # Right
                # print("Right")
                lines_r.append(line)
        else:
            intersect = find_intersection((main_a, main_b, main_c), (a, b, c))
            if intersect[1] < center[1]:
                # print("Top")
                lines_t.append(line)
            else:
                # print("Bottom")
                lines_b.append(line)

    line_l = lines_l[0]
    line_r = lines_r[0]
    line_t = lines_t[0]
    line_b = lines_b[0]
    # line_l = average_line(np.array(lines_l))
    # line_r = average_line(np.array(lines_r))
    # line_t = average_line(np.array(lines_t))
    # line_b = average_line(np.array(lines_b))

    pt_tl = find_intersection(line_l, line_t)
    pt_tr = find_intersection(line_r, line_t)
    pt_bl = find_intersection(line_l, line_b)
    pt_br = find_intersection(line_r, line_b)

    return pt_tl, pt_tr, pt_br, pt_bl


def deform_img_to_card_from_pt(
    img: np.ndarray,
    pts: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    # pts_shape: tuple[int, int] = (512, 512),
    scale_factor: float = 1,
    dst_shape: tuple[int, int] = (HIRES_WIDTH, HIRES_HEIGHT),
):
    """
    Deforms image based on given points represent corners

    Args:
        img (np.ndarray): Image array
        contour (_type_): ndarray of points, 3 dimentions
        src_shape (tuple[int, int], optional): Shape of pints. Defaults to (512, 512).
        dst_shape (tuple[int, int], optional): SHape of out image. Defaults to (600, 825).

    Returns:
        _type_: Deformed image
    """
    src_points = np.array(pts, dtype="float32") / scale_factor
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
    return cv2.warpPerspective(img, M, dst_shape, flags=cv2.INTER_CUBIC)


def deform_card(img_file: Image, output_shape: tuple[int, int] = (HIRES_WIDTH, HIRES_HEIGHT)) -> np.ndarray:
    """
    From the given image path, tries to find

    Args:
        img_path (str): Path to the image to deform

    Returns:
        np.ndarray: Deformed image or original image if it fails to find best match
    """
    # converted_img = np.flip(np.array(img_file).transpose(1, 0, 2), axis=1)
    img_file = ImageOps.exif_transpose(img_file)
    converted_img = np.array(img_file)[:, :, :3]
    # raw_img = cv2.cvtColor(converted_img, cv2.COLOR_BGR2RGB)

    loaded_shape = converted_img.shape[:2]
    max_length = max(loaded_shape)
    MAX_EDGE_LENGTH = 1200
    edge_scale_ratio = MAX_EDGE_LENGTH / max_length
    IMG_SIZE = (int(loaded_shape[1] * edge_scale_ratio), int(loaded_shape[0] * edge_scale_ratio))
    # img = resize_with_fill(cv2.cvtColor(np.array(img_file), cv2.COLOR_BGR2RGB), IMG_SIZE[0], IMG_SIZE[1])
    # img = cv2.resize(cv2.cvtColor(converted_img, cv2.COLOR_BGR2RGB), IMG_SIZE[0], IMG_SIZE[1])
    img = cv2.resize(converted_img, IMG_SIZE)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    blur_ammount = 13
    blurred_rgb = cv2.GaussianBlur(img, (blur_ammount, blur_ammount), 0)
    blurred_hsv = cv2.GaussianBlur(img_hsv, (blur_ammount, blur_ammount), 0)
    # morph_kernel = (7, 7)
    # blurred_rgb = cv2.morphologyEx(blurred_rgb, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, ksize=morph_kernel), iterations=2)
    # blurred_hsv = cv2.morphologyEx(blurred_hsv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, ksize=morph_kernel), iterations=2)
    # blurred_rgb = cv2.GaussianBlur(blurred_rgb, (blur_ammount, blur_ammount), 0)
    # blurred_hsv = cv2.GaussianBlur(blurred_hsv, (blur_ammount, blur_ammount), 0)

    alpha = 1.5
    beta = -50
    img_r = apply_contrast(blurred_rgb[:, :, 0], alpha=alpha, beta=beta)
    img_g = apply_contrast(blurred_rgb[:, :, 1], alpha=alpha, beta=beta)
    img_b = apply_contrast(blurred_rgb[:, :, 2], alpha=alpha, beta=beta)
    # img_h = apply_contrast(blurred_hsv[:, :, 0], alpha=alpha, beta=beta)
    img_s = apply_contrast(blurred_hsv[:, :, 1], alpha=alpha, beta=beta)
    img_v = apply_contrast(blurred_hsv[:, :, 2], alpha=alpha, beta=beta)
    edge_mono = cv2.cvtColor(mono_grad(blurred_rgb, 3), cv2.COLOR_BGR2GRAY)
    canney_min = 50
    canney_max = 150
    edger = expand_edges(cv2.Canny(img_r, canney_min, canney_max))
    edgeg = expand_edges(cv2.Canny(img_g, canney_min, canney_max))
    edgeb = expand_edges(cv2.Canny(img_b, canney_min, canney_max))
    # edge1 = expand_edges(cv2.Canny(img_h, canney_min, canney_max))
    edge2 = expand_edges(cv2.Canny(img_s, canney_min, canney_max))
    edge3 = expand_edges(cv2.Canny(img_v, canney_min, canney_max))

    contours_all = []
    contours, hierarchy = cv2.findContours(edger, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours_all.extend(list(remove_short_long_contours(contours, IMG_SIZE, min_length=300)))
    contours, hierarchy = cv2.findContours(edgeg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours_all.extend(list(remove_short_long_contours(contours, IMG_SIZE, min_length=300)))
    contours, hierarchy = cv2.findContours(edgeb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours_all.extend(list(remove_short_long_contours(contours, IMG_SIZE, min_length=300)))
    # contours, hierarchy = cv2.findContours(edge1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    # contours_all.extend(list(remove_short_long_contours(contours, IMG_SIZE, min_length=300)))
    contours, hierarchy = cv2.findContours(edge2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours_all.extend(list(remove_short_long_contours(contours, IMG_SIZE, min_length=300)))
    contours, hierarchy = cv2.findContours(edge3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours_all.extend(list(remove_short_long_contours(contours, IMG_SIZE, min_length=300)))
    contours, hierarchy = cv2.findContours(edge_mono, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours_all.extend(list(remove_short_long_contours(contours, IMG_SIZE, min_length=300)))

    cnt_smoothed = smooth_contours(contours_all)
    # cnt_smoothed_cleaned = remove_short_long_contours(cnt_smoothed)
    found_contours = find_rectangle_contours(cnt_smoothed, base_contour, threshold=0.3)
    if not len(found_contours) > 0:
        logger.warning("No contour was found. Exiting.")
        return cv2.resize(converted_img, output_shape)

    best_fit_contour = found_contours[0][0]
    best_fit_contour = reset_orientation(best_fit_contour)
    # best_fit_contour_convex = cv2.convexHull(best_fit_contour)
    best_fit_contour = remove_flat_points(best_fit_contour, threshold=2)
    best_fit_contour = remove_protrude(img, best_fit_contour)

    contour_image = np.zeros_like(img[:, :, 0])
    cv2.drawContours(contour_image, [best_fit_contour], -1, (255), 1)
    lines = cv2.HoughLines(contour_image, 1, np.pi / 180, 100)
    new_lines = calc_lines(lines)
    pts = get_corners_from_lines(new_lines, best_fit_contour)

    # best_fit_contour = get_corners_from_contour(best_fit_contour)
    # best_fit_contour = reset_orientation(best_fit_contour)

    return deform_img_to_card_from_pt(converted_img, pts, edge_scale_ratio, dst_shape=output_shape)
    return img, new_lines
    return contour_image


# if __name__ == "__main__":
#     img_path = "/home/yuri/github.com/AoesJP/project_pokereader/data/white_bg/IMG_1488.jpeg"
#     show_color(deform_card(img_path))
