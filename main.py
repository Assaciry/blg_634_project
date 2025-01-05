from mf3D.sfm import SfMPipeline
from mf3D.utils import get_calibration_matrix, get_image_mask_paths_zipped 

from pathlib import Path
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

def phase_1(db_path, input_path):
    if os.path.exists(db_path):
        os.remove(db_path)

    im_mask    = get_image_mask_paths_zipped(input_path)
    K = get_calibration_matrix()
    
    pipeline = SfMPipeline(db_path, image_scaling=0.25)

    for im_path,mask_path in im_mask:
        pipeline.insert_image(image_path=im_path, mask_path=mask_path, K=K)
    pipeline.match_image_pairs()

    return pipeline
    
def main():
    input_path = Path("./selected_JPG")
    db_path    = Path("./data.sqlite")
    pipeline = None

    # pipeline = phase_1(db_path, input_path)

    if pipeline is None:
        pipeline = SfMPipeline(db_path, image_scaling=0.25)
    pipeline.run_incremental_sfm()

    points = pipeline.reconstructed_points.values()
    points = np.array(list(points))
    len(points)

    # Extract x, y, z coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')

    # Add labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Point Cloud")

    plt.show()

if __name__ == "__main__":
    main()
