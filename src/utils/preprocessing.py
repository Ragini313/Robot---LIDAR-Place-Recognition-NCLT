# src/utils/preprocessing.py
import numpy as np
from typing import Optional, Tuple


def preprocess_point_cloud(
    point_cloud: np.ndarray,
    max_range: float = 80.0,
    remove_invalid: bool = True,
    downsample: Optional[int] = None
) -> np.ndarray:
    """
    Preprocess a point cloud for descriptor extraction.
    
    Args:
        point_cloud: Point cloud array (N x 3 or N x 4) [x, y, z, ...]
        max_range: Maximum range to keep points (in meters)
        remove_invalid: Whether to remove NaN and Inf values
        downsample: Downsample factor (keep every Nth point). None means no downsampling.
        
    Returns:
        Preprocessed point cloud (N' x 3) [x, y, z]
    """
    # Extract x, y, z coordinates (ignore reflectance if present)
    if point_cloud.shape[1] >= 3:
        points = point_cloud[:, :3].copy()
    else:
        raise ValueError(f"Point cloud must have at least 3 columns, got {point_cloud.shape[1]}")
    
    # Remove invalid points
    if remove_invalid:
        valid_mask = np.isfinite(points).all(axis=1)
        points = points[valid_mask]
    
    if len(points) == 0:
        return np.empty((0, 3))
    
    # Filter by range (distance from origin)
    distances = np.linalg.norm(points[:, :2], axis=1)  # 2D distance (x, y)
    range_mask = distances <= max_range
    points = points[range_mask]
    
    if len(points) == 0:
        return np.empty((0, 3))
    
    # Downsample if requested
    if downsample is not None and downsample > 1:
        points = points[::downsample]
    
    return points


def cartesian_to_polar(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates to polar coordinates.
    
    Args:
        points: Point cloud array (N x 3) [x, y, z]
        
    Returns:
        Tuple of (r, theta, z) where:
        - r: radial distance (2D distance from origin)
        - theta: azimuth angle in radians [0, 2*pi]
        - z: height (unchanged)
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Radial distance (2D)
    r = np.sqrt(x**2 + y**2)
    
    # Azimuth angle in radians [0, 2*pi]
    theta = np.arctan2(y, x)
    theta = np.mod(theta, 2 * np.pi)  # Ensure [0, 2*pi]
    
    return r, theta, z


def filter_by_height(
    point_cloud: np.ndarray,
    min_height: Optional[float] = None,
    max_height: Optional[float] = None
) -> np.ndarray:
    """
    Filter point cloud by height (z-coordinate).
    
    Args:
        point_cloud: Point cloud array (N x 3) [x, y, z]
        min_height: Minimum z value to keep
        max_height: Maximum z value to keep
        
    Returns:
        Filtered point cloud
    """
    if point_cloud.shape[1] < 3:
        raise ValueError("Point cloud must have at least 3 columns")
    
    z = point_cloud[:, 2]
    mask = np.ones(len(point_cloud), dtype=bool)
    
    if min_height is not None:
        mask &= (z >= min_height)
    
    if max_height is not None:
        mask &= (z <= max_height)
    
    return point_cloud[mask]
