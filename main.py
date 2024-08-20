# -*- coding: utf-8 -*-
"""
Created on Sat May 11 01:21:48 2024

@author: kevin
"""

import numpy as np
import cv2
import open3d as o3d
from PIL import Image
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def process_dsm_image(image_path, resize_dims=(1024, 1024), sigma_smoothing=2):
    # Load the image
    image = Image.open(image_path)
    image_array = np.array(image)
    
    # Create the 'jet' colormap and sample it
    jet_cmap = plt.get_cmap('jet')
    sampled_colors = jet_cmap(np.linspace(0, 1, 256))[:, :3]
    sampled_colors_255 = (sampled_colors * 255).astype(int)
    colormap_data = np.array(list(sampled_colors_255))
    colormap_kdtree = KDTree(colormap_data)

    # Vectorize the KDTree queries to process the entire image
    flattened_rgb_array = image_array.reshape(-1, 3)
    _, indices = colormap_kdtree.query(flattened_rgb_array)
    elevation_image = indices.reshape(image_array.shape[0], image_array.shape[1])

    # Normalize and invert the elevation image
    normalized_image = (elevation_image / elevation_image.max()) * 255
    inverted_image = 255 - normalized_image.astype(np.uint8)
    
    # Smooth the image using a Gaussian filter
    smoothed_image_array = gaussian_filter(inverted_image, sigma=sigma_smoothing)
    
    # Resize the image
    final_image = Image.fromarray(smoothed_image_array).resize(resize_dims)
    return final_image
    

def setup_interpolation():
    # Actual pixel values and their corresponding heights
    original_pixel_values = [656.725, 676.03, 678.97565]  # Example pixel values from the color scale
    original_heights = [0, 12, 20.29]  # Corresponding heights in meters

    # Maximum possible pixel value based on your information (might need adjustment)
    max_pixel_value = max(original_pixel_values)

    # Rescale pixel values to match the 0-255 scale used in image files
    scaled_pixel_values = [255 * (x / max_pixel_value) for x in original_pixel_values]

    # Create linear interpolation function with extrapolation
    return interp1d(scaled_pixel_values, original_heights, kind='linear', fill_value="extrapolate")

def pixel_to_height(gray_image, interp_func):
    # Convert RGB to grayscale assuming a simple average could work (modify if needed)
    #weighted_gray_image = 0.35 * rgb_image[:, :, 0] + 0.2 * rgb_image[:, :, 1] + 0.45 * rgb_image[:, :, 2]  # Custom weights

    
    height_image = interp_func(gray_image.astype(np.float32))
    plt.imshow(height_image, cmap='gray')
    plt.show()
    plt.close()
    return height_image

def apply_smoothing(height_image, size=25):
    # Apply a median filter for smoothing; adjust the size for more/less smoothing
    return median_filter(height_image, size=size)

def generate_point_cloud(ortho_path, dsm_path, interp_func):
    # Load the orthophoto and DSM using OpenCV
    ortho_image = cv2.imread(ortho_path)
    ortho_image = cv2.cvtColor(ortho_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    plt.imshow(ortho_image)
    plt.show()
    dsm = process_dsm_image(dsm_path)
    dsm = gaussian_filter(dsm, sigma=3)
    #dsm = histogram_equalization(dsm)
    plt.imshow(dsm, cmap='viridis')
    plt.show()
    plt.close()

    # Convert DSM RGB image to height data using the interpolation function
    dsm = pixel_to_height(dsm, interp_func)
    dsm_smoothed = apply_smoothing(dsm)  # Apply smoothing

    # Dimensions of the DSM
    height, width = dsm.shape
    
    # Prepare arrays for X, Y coordinates and colors
    x_coords = np.linspace(0, width - 1, width)
    y_coords = np.linspace(0, height - 1, height)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # Flatten the arrays
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = dsm_smoothed.flatten()  # Z-coordinates from the height map

    # Stack to create 3D points
    points = np.vstack((x_flat, y_flat, z_flat)).transpose()

    # Create color information assuming ortho_image is RGB
    colors = ortho_image.reshape(-1, 3) / 255.0  # Normalize colors

    # Create Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])

    # Save point cloud to a file
    #o3d.io.write_point_cloud("output_point_cloud.ply", point_cloud)
    return dsm

# Create interpolation function
interp_func = setup_interpolation()

# Paths to your DSM and orthophoto images
dsm = generate_point_cloud("train_15.jpg", "dsm_15.jpg", interp_func)
