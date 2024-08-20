# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 02:52:34 2024

@author: kevin
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import numpy as np
import cv2
import open3d as o3d
from PIL import Image
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
from scipy.ndimage import median_filter, gaussian_filter
import matplotlib.pyplot as plt

# Function to process DSM image
def process_dsm_image(image_path, resize_dims=(1024, 1024), sigma_smoothing=2):
    image = Image.open(image_path)
    image_array = np.array(image)
    
    jet_cmap = plt.get_cmap('jet')
    sampled_colors = jet_cmap(np.linspace(0, 1, 256))[:, :3]
    sampled_colors_255 = (sampled_colors * 255).astype(int)
    colormap_data = np.array(list(sampled_colors_255))
    colormap_kdtree = KDTree(colormap_data)

    flattened_rgb_array = image_array.reshape(-1, 3)
    _, indices = colormap_kdtree.query(flattened_rgb_array)
    elevation_image = indices.reshape(image_array.shape[0], image_array.shape[1])

    normalized_image = (elevation_image / elevation_image.max()) * 255
    inverted_image = 255 - normalized_image.astype(np.uint8)
    smoothed_image_array = gaussian_filter(inverted_image, sigma=sigma_smoothing)
    final_image = Image.fromarray(smoothed_image_array).resize(resize_dims)
    return final_image

# Setup interpolation
def setup_interpolation():
    original_pixel_values = [656.725, 676.03, 678.97565]
    original_heights = [0, 12, 20.29]
    max_pixel_value = max(original_pixel_values)
    scaled_pixel_values = [255 * (x / max_pixel_value) for x in original_pixel_values]
    return interp1d(scaled_pixel_values, original_heights, kind='linear', fill_value="extrapolate")

# Convert pixel values to height
def pixel_to_height(gray_image, interp_func):
    height_image = interp_func(gray_image.astype(np.float32))
    return height_image

# Apply smoothing
def apply_smoothing(height_image, size=25):
    return median_filter(height_image, size=size)

# Generate point cloud
def generate_point_cloud(ortho_image_path, dsm_image_path, interp_func, progress_callback=None):
    ortho_image = cv2.imread(ortho_image_path)
    ortho_image = cv2.cvtColor(ortho_image, cv2.COLOR_BGR2RGB)
    
    dsm = process_dsm_image(dsm_image_path)
    
    dsm = gaussian_filter(dsm, sigma=3)
    dsm = pixel_to_height(dsm, interp_func)
    
    dsm_smoothed = apply_smoothing(dsm)
    
    if progress_callback:
        progress_callback(90)  # Update progress
    
    height, width = dsm.shape
    x_coords = np.linspace(0, width - 1, width)
    y_coords = np.linspace(0, height - 1, height)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = dsm_smoothed.flatten()
    
    points = np.vstack((x_flat, y_flat, z_flat)).transpose()
    colors = ortho_image.reshape(-1, 3) / 255.0
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([point_cloud])
    
    return point_cloud

def save_point_cloud(point_cloud, filepath):
    o3d.io.write_point_cloud(filepath, point_cloud)

# Function to handle file selection
def select_original_image():
    filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png")])
    if filepath:
        original_image_path.set(filepath)

def select_dsm_image():
    filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png")])
    if filepath:
        dsm_image_path.set(filepath)

def update_progress(value):
    progress_bar["value"] = value
    progress_bar.update()

def generate_mesh():
    if not original_image_path.get() or not dsm_image_path.get():
        messagebox.showwarning("Input Error", "Please select both the original and DSM images.")
        return
    
    def run_mesh_generation():
        #progress_bar.start()
        interp_func = setup_interpolation()
        update_progress(20)
        global generated_point_cloud
        generated_point_cloud = generate_point_cloud(original_image_path.get(), dsm_image_path.get(), interp_func, progress_callback=update_progress)
        update_progress(100)
        #progress_bar.stop()
        status_bar.config(text="3D Mesh generated. Ready to download.")

        download_button.config(state="normal")
        progress_bar["value"] = 0  # Reset the progress bar

    # Run the mesh generation in a separate thread
    threading.Thread(target=run_mesh_generation).start()

def download_mesh():
    if generated_point_cloud is None:
        messagebox.showwarning("No Mesh Generated", "Please generate a 3D mesh before downloading.")
        return

    filepath = filedialog.asksaveasfilename(defaultextension=".ply", filetypes=[("PLY Files", "*.ply")])
    if filepath:
        save_point_cloud(generated_point_cloud, filepath)
        status_bar.config(text=f"Mesh saved as {filepath}")
        messagebox.showinfo("Success", f"3D Mesh successfully saved as {filepath}")

# Tkinter setup
root = tk.Tk()
root.title("3D Mesh Generator")

# Set the window to open at a size relative to the screen size
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Open the app at 70% of the screen size
app_width = int(screen_width * 0.7)
app_height = int(screen_height * 0.7)

# Set the geometry of the window
root.geometry(f"{app_width}x{app_height}")

# Make sure the window opens in a normal state (not minimized)
root.state('normal')

# Styles and themes
style = ttk.Style()
style.theme_use('clam')

# Define color palette
primary_color = "#2c3e50"  # Dark Blue
secondary_color = "#ecf0f1"  # Light Gray
accent_color = "#27ae60"  # Green
text_color = "#ecf0f1"  # Light Gray (for text)

style.configure("TButton", font=("Arial", 10), padding=6, background=primary_color, foreground=text_color)
style.configure("TLabel", font=("Arial", 10), background=primary_color, foreground=text_color)
style.configure("TEntry", font=("Arial", 10), fieldbackground=secondary_color, foreground=primary_color)
style.configure("TFrame", background=primary_color)

# Customize progress bar
style.configure("Green.Horizontal.TProgressbar", troughcolor=secondary_color, background=accent_color, thickness=20)

# Variables
original_image_path = tk.StringVar()
dsm_image_path = tk.StringVar()
generated_point_cloud = None

# Menu bar
menu_bar = tk.Menu(root, bg=primary_color, fg=text_color)
root.config(menu=menu_bar)

help_menu = tk.Menu(menu_bar, tearoff=0, bg=primary_color, fg=text_color)
menu_bar.add_cascade(label="Help", menu=help_menu)
help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "3D Mesh Generator v1.0"))

# Main layout
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

ttk.Label(frame, text="Original Image:").grid(row=0, column=0, padx=10, pady=10)
ttk.Entry(frame, textvariable=original_image_path, width=50).grid(row=0, column=1, padx=10, pady=10, sticky=(tk.W, tk.E))
ttk.Button(frame, text="Browse...", command=select_original_image).grid(row=0, column=2, padx=10, pady=10)

ttk.Label(frame, text="DSM Image:").grid(row=1, column=0, padx=10, pady=10)
ttk.Entry(frame, textvariable=dsm_image_path, width=50).grid(row=1, column=1, padx=10, pady=10, sticky=(tk.W, tk.E))
ttk.Button(frame, text="Browse...", command=select_dsm_image).grid(row=1, column=2, padx=10, pady=10)

ttk.Button(frame, text="Generate 3D Mesh", command=generate_mesh, width=20).grid(row=2, column=0, columnspan=3, pady=20)

# Progress bar
progress_bar = ttk.Progressbar(frame, style="Green.Horizontal.TProgressbar", mode='determinate', maximum=100)
progress_bar.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky=(tk.W, tk.E))

# Download button
download_button = ttk.Button(frame, text="Download 3D Mesh", command=download_mesh, width=20, state="disabled")
download_button.grid(row=4, column=0, columnspan=3, pady=20)

# Status bar
status_bar = ttk.Label(root, text="Ready", relief=tk.SUNKEN, anchor=tk.W, padding=5, background=primary_color, foreground=text_color)
status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))

# Allow resizing
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
frame.columnconfigure(1, weight=1)

root.mainloop()
