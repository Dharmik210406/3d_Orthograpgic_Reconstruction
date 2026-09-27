import os
import sys
import time
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image
from scipy.ndimage import (
    binary_erosion,
    binary_dilation,
    binary_fill_holes,
    gaussian_filter,
    label,
)

# Optional OpenCV support with clean native fallback
try:
    HAS_OPENCV = importlib.util.find_spec("cv2") is not None
except ImportError:
    HAS_OPENCV = False

if HAS_OPENCV:
    import cv2  # type: ignore[import-not-found]


# Configuration & Paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TOP_IMAGE_PATH = os.path.join(BASE_DIR, "top_view.png")
FRONT_IMAGE_PATH = os.path.join(BASE_DIR, "front_view.png")
SIDE_IMAGE_PATH = os.path.join(BASE_DIR, "side_view.png")
OUTPUT_DIRECTORY = BASE_DIR

# Reconstruction Parameters
VOXEL_RESOLUTION = 96
MASK_ALPHA = 0.45
CAMERA_ELEVATION = 30
CAMERA_AZIMUTH = 45
EXTRACT_SURFACE_ONLY = True
MAX_VOXELS_FOR_RENDER = 150_000



# Helper & Utility Functions


def log_step(message: str) -> None:
    """Prints a timestamped status update."""
    current_time = time.strftime('%H:%M:%S')
    print(f"[{current_time}] {message}")


def generate_synthetic_sketch(view_type: str, size=(400, 400)) -> np.ndarray:
    """Generates a synthetic orthographic sketch if user images are not provided."""
    canvas = np.ones(size, dtype=np.uint8) * 255
    cx, cy = size[0] // 2, size[1] // 2
    y, x = np.ogrid[:size[0], :size[1]]

    if view_type == 'top':
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= 110 ** 2
    elif view_type == 'front':
        mask = (np.abs(x - cx) <= 80) & (np.abs(y - cy) <= 120)
    else:  # side
        mask = (np.abs(x - cx) <= 70) & (y >= cy - 120) & (y <= cy + 120) & ((x - cx) + (y - cy) <= 100)

    eroded = binary_erosion(mask, iterations=4)
    boundary = mask & (~eroded)
    canvas[boundary] = 0
    return canvas


def load_grayscale_image(filepath: str, fallback_view: str) -> np.ndarray:
    """Loads image in grayscale, providing a fallback if the file is absent."""
    if not os.path.exists(filepath):
        log_step(f"Notice: '{os.path.basename(filepath)}' not found. Using template sketch.")
        sample = generate_synthetic_sketch(fallback_view)
        save_image(filepath, sample)
        return sample

    try:
        pil_img = Image.open(filepath).convert('L')
        return np.array(pil_img, dtype=np.uint8)
    except Exception as exc:
        log_step(f"Warning: Failed to read {filepath} ({exc}). Using template sketch.")
        return generate_synthetic_sketch(fallback_view)


def save_image(filepath: str, img_array: np.ndarray) -> None:
    """Saves a NumPy image array using Pillow."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)
    Image.fromarray(img_array).save(filepath)


def resize_image(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resizes a 2D image to target dimensions."""
    pil_img = Image.fromarray(image)
    return np.array(pil_img.resize(target_size, resample=Image.Resampling.NEAREST))


def pad_image_to_square(image: np.ndarray, fill_value=255) -> np.ndarray:
    """Symmetrically pads an image to a square canvas preserving aspect ratio."""
    height, width = image.shape[:2]
    max_dim = max(height, width)
    pad_top = (max_dim - height) // 2
    pad_bottom = max_dim - height - pad_top
    pad_left = (max_dim - width) // 2
    pad_right = max_dim - width - pad_left

    if image.ndim == 2:
        return np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)),
                      mode='constant', constant_values=fill_value)
    return np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                  mode='constant', constant_values=fill_value)


def crop_to_content(binary_mask: np.ndarray) -> np.ndarray:
    """Crops tight bounding box around the active silhouette."""
    y_coords, x_coords = np.where(binary_mask > 0)
    if len(x_coords) == 0:
        return binary_mask

    y_min = max(y_coords.min() - 5, 0)
    x_min = max(x_coords.min() - 5, 0)
    y_max = min(y_coords.max() + 5, binary_mask.shape[0] - 1)
    x_max = min(x_coords.max() + 5, binary_mask.shape[1] - 1)
    return binary_mask[y_min:y_max + 1, x_min:x_max + 1]



# Image Processing & Feature Extraction


def extract_silhouette(gray_image: np.ndarray) -> np.ndarray:
    """
    Converts a line sketch into a solid filled binary silhouette.
    Uses pure SciPy/NumPy morphology or OpenCV when available.
    """
    if HAS_OPENCV:
        norm = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
        thresh = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 10)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1),
                             kernel, iterations=2)
        inv = 255 - dilated
        flood = inv.copy()
        h, w = flood.shape
        cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 0)
        solid = (flood > 0).astype(np.uint8) * 255
    else:
        # High quality pure SciPy adaptive threshold and morphological hole-filling
        blurred = gaussian_filter(gray_image.astype(float), sigma=2.0)
        lines = gray_image < (blurred - 8)
        dilated = binary_dilation(lines, iterations=2)
        solid = binary_fill_holes(dilated).astype(np.uint8) * 255

    cropped = crop_to_content(solid)
    padded = pad_image_to_square(cropped, fill_value=0)
    resized = resize_image(padded, (VOXEL_RESOLUTION, VOXEL_RESOLUTION))
    return resized > 0


def create_overlay(original_gray: np.ndarray, mask: np.ndarray, alpha=0.45) -> np.ndarray:
    """Blends a colored mask layer over the original sketch for visual validation."""
    squared = resize_image(pad_image_to_square(original_gray, 255), (VOXEL_RESOLUTION, VOXEL_RESOLUTION))
    rgb_image = np.stack([squared] * 3, axis=-1)

    tint_layer = np.zeros_like(rgb_image)
    tint_layer[:, :, 0] = 0
    tint_layer[:, :, 1] = 255
    tint_layer[:, :, 2] = 255  # Cyan tint

    output = rgb_image.copy()
    output[mask] = (output[mask] * (1 - alpha) + tint_layer[mask] * alpha).astype(np.uint8)
    return output



# 3D Reconstruction & Carving Logic


def carve_voxels(top_view: np.ndarray, front_view: np.ndarray, side_view: np.ndarray) -> np.ndarray:
    """
    Computes visual hull via 3D tensor intersection (logical AND).
    Broadcasting dimensions:
      top_view[None, :, :]   -> (1, Y, X)
      front_view[:, None, :] -> (Z, 1, X)
      side_view[:, :, None]  -> (Z, Y, 1)
    """
    return top_view[None, :, :] & front_view[:, None, :] & side_view[:, :, None]


def extract_surface_voxels(voxel_grid: np.ndarray) -> np.ndarray:
    """Removes interior voxels to yield an optimized surface shell."""
    if not np.any(voxel_grid):
        return voxel_grid
    eroded = binary_erosion(voxel_grid, iterations=1)
    shell = voxel_grid & (~eroded)
    return shell if np.any(shell) else voxel_grid


def downsample_for_rendering(voxel_grid: np.ndarray) -> tuple:
    """Adjusts stride dynamically to fit rendering budget."""
    total = int(voxel_grid.sum())
    if total <= MAX_VOXELS_FOR_RENDER:
        return voxel_grid, 1

    stride = 2
    while True:
        downsampled = voxel_grid[::stride, ::stride, ::stride]
        if int(downsampled.sum()) <= MAX_VOXELS_FOR_RENDER or stride >= 4:
            return downsampled, stride
        stride += 1


def find_largest_component_volume(voxel_grid: np.ndarray) -> int:
    """Returns size of the primary connected 3D component to filter noise."""
    labels, count = label(voxel_grid)
    if count == 0:
        return 0
    return int(np.bincount(labels.ravel())[1:].max())


def optimize_orientations(top_mask: np.ndarray, front_mask: np.ndarray, side_mask: np.ndarray) -> tuple:
    """Evaluates chirality flips to ensure optimal view alignment."""
    candidates = []
    flips = [(False, False), (True, False), (False, True), (True, True)]

    for flip_front, flip_side in flips:
        t_front = np.fliplr(front_mask) if flip_front else front_mask
        t_side = np.fliplr(side_mask) if flip_side else side_mask
        model = carve_voxels(top_mask, t_front, t_side)
        score = (find_largest_component_volume(model), model.sum())
        candidates.append((score, t_front, t_side, model))

    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0]
    return best[1], best[2], best[3]



# Visualizations & Dashboard


def render_3d_model(voxel_grid: np.ndarray, elevation: int, azimuth: int,
                    alpha=0.95, save_path=None, title="Isometric View") -> None:
    """Renders 3D voxel model with Matplotlib and saves high-resolution figure."""
    z_dim, y_dim, x_dim = voxel_grid.shape
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    voxels_xyz = np.transpose(voxel_grid, (2, 1, 0))
    ax.voxels(voxels_xyz, facecolors=None, edgecolor=None, linewidth=0.0, alpha=alpha)

    ax.set_xlim(0, x_dim)
    ax.set_ylim(0, y_dim)
    ax.set_zlim(0, z_dim)
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_axis_off()
    ax.set_title(title, pad=8)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_summary_dashboard(orig_top: np.ndarray, orig_front: np.ndarray, orig_side: np.ndarray,
                               mask_top: np.ndarray, mask_front: np.ndarray, mask_side: np.ndarray,
                               voxel_model: np.ndarray, save_path: str) -> None:
    """Generates a complete multi-panel dashboard comparing inputs, masks, and 3D output."""
    fig = plt.figure(figsize=(14, 7))
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.25], wspace=0.05, hspace=0.08)

    inputs = [(orig_top, "Top View (Input)"), (orig_front, "Front View (Input)"), (orig_side, "Side View (Input)")]
    for idx, (img, title) in enumerate(inputs):
        ax = fig.add_subplot(grid[0, idx])
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    overlays = [(mask_top, "Top Mask Overlay"), (mask_front, "Front Mask Overlay"), (mask_side, "Side Mask Overlay")]
    for idx, (img, title) in enumerate(overlays):
        ax = fig.add_subplot(grid[1, idx])
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    ax_3d = fig.add_subplot(grid[:, 3], projection='3d')
    voxels_xyz = np.transpose(voxel_model, (2, 1, 0))
    ax_3d.voxels(voxels_xyz, facecolors=None, edgecolor=None, linewidth=0.0, alpha=0.95)

    z_dim, y_dim, x_dim = voxel_model.shape
    ax_3d.set_xlim(0, x_dim)
    ax_3d.set_ylim(0, y_dim)
    ax_3d.set_zlim(0, z_dim)
    ax_3d.view_init(CAMERA_ELEVATION, CAMERA_AZIMUTH)
    ax_3d.set_axis_off()
    ax_3d.set_title("Reconstructed 3D Voxel Model", pad=10)

    fig.suptitle("Multi-View Orthographic to 3D Voxel Reconstruction", fontsize=12, y=0.99)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)



# Main Pipeline


def main() -> None:
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    log_step("Starting 3D reconstruction pipeline...")

    log_step("Loading multi-view orthographic inputs...")
    top_img = load_grayscale_image(TOP_IMAGE_PATH, fallback_view='top')
    front_img = load_grayscale_image(FRONT_IMAGE_PATH, fallback_view='front')
    side_img = load_grayscale_image(SIDE_IMAGE_PATH, fallback_view='side')

    log_step("Extracting solid binary silhouettes from line sketches...")
    top_mask = extract_silhouette(top_img)
    front_mask = extract_silhouette(front_img)
    side_mask = extract_silhouette(side_img)

    log_step("Optimizing orientation chirality and carving visual hull...")
    front_mask, side_mask, voxel_model = optimize_orientations(top_mask, front_mask, side_mask)

    if EXTRACT_SURFACE_ONLY:
        log_step("Extracting outer surface shell for optimized rendering...")
        voxel_model = extract_surface_voxels(voxel_model)

    log_step(f"Total occupied voxels: {int(voxel_model.sum()):,}")

    log_step("Evaluating rendering budget and downsampling if required...")
    render_model, stride = downsample_for_rendering(voxel_model)
    if stride > 1:
        log_step(f"Downsampled with stride {stride}. Active render voxels: ~{int(render_model.sum()):,}")

    log_step("Generating colored diagnostic overlays...")
    top_overlay = create_overlay(top_img, top_mask, MASK_ALPHA)
    front_overlay = create_overlay(front_img, front_mask, MASK_ALPHA)
    side_overlay = create_overlay(side_img, side_mask, MASK_ALPHA)

    iso_out_path = os.path.join(OUTPUT_DIRECTORY, "isometric_view.png")
    summary_out_path = os.path.join(OUTPUT_DIRECTORY, "reconstruction_summary.png")

    log_step("Rendering final 3D isometric perspective...")
    render_3d_model(render_model, elevation=CAMERA_ELEVATION, azimuth=CAMERA_AZIMUTH,
                    alpha=0.95, save_path=iso_out_path, title="Isometric 3D View")

    log_step("Compiling multi-panel summary dashboard...")
    top_disp = resize_image(pad_image_to_square(top_img, 255), (VOXEL_RESOLUTION, VOXEL_RESOLUTION))
    front_disp = resize_image(pad_image_to_square(front_img, 255), (VOXEL_RESOLUTION, VOXEL_RESOLUTION))
    side_disp = resize_image(pad_image_to_square(side_img, 255), (VOXEL_RESOLUTION, VOXEL_RESOLUTION))

    generate_summary_dashboard(top_disp, front_disp, side_disp,
                               top_overlay, front_overlay, side_overlay,
                               render_model, summary_out_path)

    log_step("Saving intermediate silhouette masks and overlays...")
    save_image(os.path.join(OUTPUT_DIRECTORY, "mask_top.png"), (top_mask * 255).astype(np.uint8))
    save_image(os.path.join(OUTPUT_DIRECTORY, "mask_front.png"), (front_mask * 255).astype(np.uint8))
    save_image(os.path.join(OUTPUT_DIRECTORY, "mask_side.png"), (side_mask * 255).astype(np.uint8))
    save_image(os.path.join(OUTPUT_DIRECTORY, "overlay_top.png"), top_overlay)
    save_image(os.path.join(OUTPUT_DIRECTORY, "overlay_front.png"), front_overlay)
    save_image(os.path.join(OUTPUT_DIRECTORY, "overlay_side.png"), side_overlay)

    log_step("Execution completed successfully!")
    print(f" -> 3D Isometric View:      {iso_out_path}")
    print(f" -> Summary Dashboard:      {summary_out_path}")


if __name__ == "__main__":
    main()

