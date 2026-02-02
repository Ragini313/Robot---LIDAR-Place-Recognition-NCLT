# src/utils/data_loader.py
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import yaml

class NCLTDataLoader:
    """Loader for NCLT dataset LiDAR and ground truth data."""
    
    def __init__(self, base_path: str):
        """
        Initialize the NCLT data loader.
        
        Args:
            base_path: Path to the root directory containing NCLT data
        """
        self.base_path = Path(base_path)
        self.sessions = ['2012-01-08', '2012-01-15', '2012-01-22']
        
    def load_ground_truth(self, session: str) -> pd.DataFrame:
        """
        Load ground truth poses for a session.
        
        Args:
            session: Session name (e.g., '2012-01-08')
            
        Returns:
            DataFrame with ground truth poses
        """
        # Try groundtruth_YYYY-MM-DD.csv format first
        gt_path = self.base_path / session / f'groundtruth_{session}.csv'
        if not gt_path.exists():
            # Fallback to groundtruth.csv
            gt_path = self.base_path / session / 'groundtruth.csv'
        
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        
        # Load ground truth data
        # Format: timestamp, x, y, z, roll, pitch, yaw
        gt_data = pd.read_csv(gt_path, header=None, low_memory=False)
        gt_data.columns = ['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']
        
        # Convert all columns to numeric, forcing errors to NaN, then drop them
        for col in gt_data.columns:
            gt_data[col] = pd.to_numeric(gt_data[col], errors='coerce')
        
        # Drop rows with NaN in essential columns
        gt_data = gt_data.dropna(subset=['x', 'y', 'timestamp'])
        
        return gt_data
    
    def get_velodyne_files(self, session: str) -> List[Path]:
        """
        Get list of Velodyne scan files for a session.
        
        Args:
            session: Session name
            
        Returns:
            List of Path objects to .bin files
        """
        # Try NCLT structure: YYYY-MM-DD_vel/YYYY-MM-DD/velodyne_sync/
        velodyne_path = self.base_path / session / f'{session}_vel' / session / 'velodyne_sync'
        if not velodyne_path.exists():
            # Try alternative: YYYY-MM-DD_vel/YYYY-MM-DD/
            velodyne_path = self.base_path / session / f'{session}_vel' / session
        if not velodyne_path.exists():
            # Try simple structure: velodyne_data
            velodyne_path = self.base_path / session / 'velodyne_data'
        if not velodyne_path.exists():
            # Try: velodyne
            velodyne_path = self.base_path / session / 'velodyne'
        
        if not velodyne_path.exists():
            raise FileNotFoundError(f"Velodyne directory not found. Tried: {velodyne_path}")
        
        # Get all .bin files recursively
        bin_files = sorted(list(velodyne_path.rglob('*.bin')))
        
        if not bin_files:
            raise FileNotFoundError(f"No .bin files found in {velodyne_path}")
        
        return bin_files
    
    def load_velodyne_scan(self, filepath: Path) -> np.ndarray:
        """
        Load a single Velodyne scan from .bin file.
        
        NCLT format: Each point is 10 bytes:
        - x (2 bytes, uint16)
        - y (2 bytes, uint16)
        - z (2 bytes, uint16)
        - intensity (1 byte, uint8)
        - laser_id (1 byte, uint8)
        - utime (2 bytes, uint16)
        
        Args:
            filepath: Path to .bin file
            
        Returns:
            Numpy array of point cloud (N x 3) [x, y, z]
        """
        # Read as uint16 (2 bytes each)
        raw_data = np.fromfile(filepath, dtype=np.uint16)
        
        # Each point is 10 bytes = 5 uint16 elements
        num_points = len(raw_data) // 5
        if num_points == 0:
            return np.empty((0, 3))
        
        raw_data = raw_data[:num_points * 5].reshape(-1, 5)
        
        # NCLT coordinates are scaled: (value * 0.005) - 100.0
        x = raw_data[:, 0].astype(np.float32) * 0.005 - 100.0
        y = raw_data[:, 1].astype(np.float32) * 0.005 - 100.0
        z = raw_data[:, 2].astype(np.float32) * 0.005 - 100.0
        
        # Return as N x 3 array
        return np.column_stack((x, y, z))
    
    def extract_timestamp_from_filename(self, filepath: Path) -> Optional[int]:
        """
        Extract timestamp from Velodyne filename.
        NCLT filenames are typically: timestamp.bin
        
        Args:
            filepath: Path to .bin file
            
        Returns:
            Timestamp as integer, or None if extraction fails
        """
        try:
            # Filename without extension should be the timestamp
            timestamp_str = filepath.stem
            return int(timestamp_str)
        except (ValueError, AttributeError):
            return None
    
    def synchronize_scans_with_poses(self, velodyne_files: List[Path], gt_data: pd.DataFrame) -> List[Tuple[Path, pd.Series]]:
        """
        Synchronize Velodyne scans with ground truth poses using timestamps.
        
        Args:
            velodyne_files: List of Velodyne scan file paths
            gt_data: DataFrame with ground truth poses
            
        Returns:
            List of tuples (filepath, pose_row) for synchronized scans
        """
        synchronized = []
        
        for vel_file in velodyne_files:
            timestamp = self.extract_timestamp_from_filename(vel_file)
            if timestamp is None:
                continue
            
            # Find closest ground truth pose by timestamp
            time_diffs = np.abs(gt_data['timestamp'] - timestamp)
            closest_idx = time_diffs.idxmin()
            closest_pose = gt_data.loc[closest_idx]
            
            # Only include if timestamp difference is reasonable
            # NCLT timestamps are in microseconds, so 0.1 seconds = 100000 microseconds
            # Use a more lenient threshold: 0.5 seconds = 500000 microseconds
            if time_diffs[closest_idx] < 500000:  # 0.5 seconds in microseconds
                synchronized.append((vel_file, closest_pose))
        
        return synchronized
    
    def load_session_data(self, session: str, max_scans: Optional[int] = None, downsample_rate: int = 1) -> Dict:
        """
        Load all data for a session.
        
        Args:
            session: Session name
            max_scans: Maximum number of scans to load (for testing)
            downsample_rate: Use every Nth scan (default: 1, use all scans)
            
        Returns:
            Dictionary with session data including synchronized scans and poses
        """
        print(f"Loading session: {session}")
        
        # Load ground truth
        gt_data = self.load_ground_truth(session)
        
        # Get Velodyne files
        velodyne_files = self.get_velodyne_files(session)
        
        # Apply downsampling
        if downsample_rate > 1:
            velodyne_files = velodyne_files[::downsample_rate]
        
        if max_scans:
            velodyne_files = velodyne_files[:max_scans]
        
        print(f"Found {len(velodyne_files)} Velodyne scans")
        
        # Synchronize scans with ground truth poses
        print("Synchronizing scans with ground truth poses...")
        synchronized_scans = self.synchronize_scans_with_poses(velodyne_files, gt_data)
        print(f"Synchronized {len(synchronized_scans)} scans with poses")
        
        session_data = {
            'session': session,
            'ground_truth': gt_data,
            'velodyne_files': velodyne_files,
            'synchronized_scans': synchronized_scans,  # List of (filepath, pose) tuples
            'num_scans': len(velodyne_files),
            'num_synchronized': len(synchronized_scans)
        }
        
        return session_data
