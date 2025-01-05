from pathlib import Path
import os
import numpy as np

def get_calibration_matrix():
    # Constants from metadata
    focal_length_mm = 35.0  # Focal length in mm
    x_resolution = 5798.66  # Pixels per inch
    y_resolution = 5788.94  # Pixels per inch
    image_width = 5184
    image_height = 3456

    # Conversion factor (1 inch = 25.4 mm)
    pixels_per_mm_x = x_resolution / 25.4
    pixels_per_mm_y = y_resolution / 25.4

    # Focal length in pixels
    f_x = focal_length_mm * pixels_per_mm_x
    f_y = focal_length_mm * pixels_per_mm_y

    # Principal point (assuming image center)
    c_x = image_width / 2
    c_y = image_height / 2

    # Intrinsic matrix K
    K = np.array([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0,  0,  1]
    ])

    return K

def get_image_mask_paths_zipped(base_path : Path):
    contents = os.listdir(base_path)
    im_names = [c for c in contents if len(c.split("_")) == 2]
    mask_names = [c.split(".")[0] + "_label.png" for c in im_names]
    return list(zip([base_path/im for im in im_names],[base_path/lab for lab in mask_names]))