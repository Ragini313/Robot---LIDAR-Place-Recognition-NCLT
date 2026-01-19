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
        gt_path = self.base_path / session / 'groundtruth.csv'
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        
        # Load ground truth data
        # Format: timestamp, x, y, z, roll, pitch, yaw
        gt_data = pd.read_csv(gt_path, header=None)
        gt_data.columns = ['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']
        
        return gt_data
    
    def get_velodyne_files(self, session: str) -> List[Path]:
        """
        Get list of Velodyne scan files for a session.
        
        Args:
            session: Session name
            
        Returns:
            List of Path objects to .bin files
        """
        velodyne_path = self.base_path / session / 'velodyne_data'
        if not velodyne_path.exists():
            # Try alternative path structure
            velodyne_path = self.base_path / session / 'velodyne'
        
        if not velodyne_path.exists():
            raise FileNotFoundError(f"Velodyne directory not found: {velodyne_path}")
        
        # Get all .bin files
        bin_files = sorted(list(velodyne_path.glob('*.bin')))
        
        return bin_files
    
    def load_velodyne_scan(self, filepath: Path) -> np.ndarray:
        """
        Load a single Velodyne scan from .bin file.
        
        Args:
            filepath: Path to .bin file
            
        Returns:
            Numpy array of point cloud (N x 4) [x, y, z, reflectance]
        """
        # NCLT Velodyne data format: 32-bit floats
        scan = np.fromfile(filepath, dtype=np.float32)
        # Reshape to N x 4 (x, y, z, reflectance)
        scan = scan.reshape(-1, 4)
        
        return scan
    
    def load_session_data(self, session: str, max_scans: Optional[int] = None) -> Dict:
        """
        Load all data for a session.
        
        Args:
            session: Session name
            max_scans: Maximum number of scans to load (for testing)
            
        Returns:
            Dictionary with session data
        """
        print(f"Loading session: {session}")
        
        # Load ground truth
        gt_data = self.load_ground_truth(session)
        
        # Get Velodyne files
        velodyne_files = self.get_velodyne_files(session)
        
        if max_scans:
            velodyne_files = velodyne_files[:max_scans]
        
        print(f"Found {len(velodyne_files)} Velodyne scans")
        
        session_data = {
            'session': session,
            'ground_truth': gt_data,
            'velodyne_files': velodyne_files,
            'num_scans': len(velodyne_files)
        }
        
        return session_data
