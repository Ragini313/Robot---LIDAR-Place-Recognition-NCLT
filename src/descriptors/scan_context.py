# src/descriptors/scan_context.py
import numpy as np
from typing import Optional, Tuple
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.preprocessing import preprocess_point_cloud, cartesian_to_polar


class ScanContextDescriptor:
    """
    ScanContext descriptor for LiDAR place recognition.
    
    Divides point cloud into rings (radial bins) and sectors (angular bins),
    then computes height features for each bin.
    """
    
    def __init__(
        self,
        num_sectors: int = 60,
        num_rings: int = 20,
        max_range: float = 80.0,
        height_feature: str = 'max'
    ):
        """
        Initialize ScanContext descriptor.
        
        Args:
            num_sectors: Number of angular sectors (default: 60)
            num_rings: Number of radial rings (default: 20)
            max_range: Maximum range in meters (default: 80.0)
            height_feature: Height feature type: 'max', 'mean', 'min' (default: 'max')
        """
        self.num_sectors = num_sectors
        self.num_rings = num_rings
        self.max_range = max_range
        self.height_feature = height_feature
        
        # Precompute ring boundaries
        self.ring_edges = np.linspace(0, max_range, num_rings + 1)
        # Sector boundaries (angular)
        self.sector_edges = np.linspace(0, 2 * np.pi, num_sectors + 1)
    
    def compute(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Compute ScanContext descriptor from point cloud.
        
        Args:
            point_cloud: Point cloud array (N x 3 or N x 4) [x, y, z, ...]
            
        Returns:
            ScanContext matrix (num_rings x num_sectors)
        """
        # Preprocess point cloud
        points = preprocess_point_cloud(point_cloud, max_range=self.max_range)
        
        if len(points) == 0:
            # Return zero matrix if no points
            return np.zeros((self.num_rings, self.num_sectors))
        
        # Convert to polar coordinates
        r, theta, z = cartesian_to_polar(points)
        
        # Initialize descriptor matrix
        descriptor = np.zeros((self.num_rings, self.num_sectors))
        
        # For mean feature, track counts
        if self.height_feature == 'mean':
            counts = np.zeros((self.num_rings, self.num_sectors))
        
        # Bin points into rings and sectors
        for i, point in enumerate(points):
            # Find ring index
            ring_idx = np.digitize(r[i], self.ring_edges) - 1
            ring_idx = np.clip(ring_idx, 0, self.num_rings - 1)
            
            # Find sector index
            sector_idx = np.digitize(theta[i], self.sector_edges) - 1
            sector_idx = np.clip(sector_idx, 0, self.num_sectors - 1)
            
            # Update height feature based on type
            if self.height_feature == 'max':
                descriptor[ring_idx, sector_idx] = max(
                    descriptor[ring_idx, sector_idx],
                    z[i]
                )
            elif self.height_feature == 'min':
                if descriptor[ring_idx, sector_idx] == 0:
                    descriptor[ring_idx, sector_idx] = z[i]
                else:
                    descriptor[ring_idx, sector_idx] = min(
                        descriptor[ring_idx, sector_idx],
                        z[i]
                    )
            elif self.height_feature == 'mean':
                descriptor[ring_idx, sector_idx] += z[i]
                counts[ring_idx, sector_idx] += 1
        
        # Normalize mean features
        if self.height_feature == 'mean':
            # Compute mean by dividing sum by count (avoid division by zero)
            mask = counts > 0
            descriptor[mask] = descriptor[mask] / counts[mask]
        
        return descriptor
    
    def compute_ring_key(self, descriptor: np.ndarray) -> np.ndarray:
        """
        Compute ring key for fast retrieval.
        Ring key is the maximum value in each ring.
        
        Args:
            descriptor: ScanContext matrix (num_rings x num_sectors)
            
        Returns:
            Ring key vector (num_rings,)
        """
        return np.max(descriptor, axis=1)
    
    def compute_sector_key(self, descriptor: np.ndarray) -> np.ndarray:
        """
        Compute sector key for fast retrieval.
        Sector key is the maximum value in each sector.
        
        Args:
            descriptor: ScanContext matrix (num_rings x num_sectors)
            
        Returns:
            Sector key vector (num_sectors,)
        """
        return np.max(descriptor, axis=0)
    
    def match_with_rotation(
        self,
        descriptor1: np.ndarray,
        descriptor2: np.ndarray,
        similarity_metric: str = 'cosine'
    ) -> Tuple[float, int]:
        """
        Match two descriptors with rotation invariance (circular shift).
        
        Args:
            descriptor1: First ScanContext matrix
            descriptor2: Second ScanContext matrix
            similarity_metric: 'cosine', 'euclidean', or 'l1'
            
        Returns:
            Tuple of (best_similarity, best_shift) where best_shift is the
            circular shift that gives the best match
        """
        best_similarity = -np.inf if similarity_metric != 'euclidean' and similarity_metric != 'l1' else np.inf
        best_shift = 0
        
        # Try all circular shifts
        for shift in range(self.num_sectors):
            # Circular shift descriptor2
            shifted_desc2 = np.roll(descriptor2, shift, axis=1)
            
            # Compute similarity
            if similarity_metric == 'cosine':
                # Flatten and compute cosine similarity
                vec1 = descriptor1.flatten()
                vec2 = shifted_desc2.flatten()
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 > 0 and norm2 > 0:
                    similarity = dot_product / (norm1 * norm2)
                else:
                    similarity = 0.0
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_shift = shift
                    
            elif similarity_metric == 'euclidean':
                diff = descriptor1 - shifted_desc2
                distance = np.linalg.norm(diff)
                if distance < best_similarity:
                    best_similarity = distance
                    best_shift = shift
                    
            elif similarity_metric == 'l1':
                diff = descriptor1 - shifted_desc2
                distance = np.sum(np.abs(diff))
                if distance < best_similarity:
                    best_similarity = distance
                    best_shift = shift
        
        # For distance metrics, convert to similarity (negative distance)
        if similarity_metric in ['euclidean', 'l1']:
            best_similarity = -best_similarity  # Negative distance as similarity
        
        return best_similarity, best_shift
