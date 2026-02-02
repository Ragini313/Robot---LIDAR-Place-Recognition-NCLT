# src/matching/retrieval.py
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from descriptors.scan_context import ScanContextDescriptor


class PlaceRetrievalSystem:
    """
    Place retrieval system for matching LiDAR scan descriptors.
    """
    
    def __init__(
        self,
        descriptor_type: str = 'ScanContext',
        similarity_metric: str = 'cosine',
        descriptor_params: Optional[Dict] = None
    ):
        """
        Initialize the retrieval system.
        
        Args:
            descriptor_type: Type of descriptor ('ScanContext')
            similarity_metric: Similarity metric ('cosine', 'euclidean', 'l1')
            descriptor_params: Parameters for descriptor initialization
        """
        self.similarity_metric = similarity_metric
        self.descriptor_type = descriptor_type
        
        # Initialize descriptor extractor
        if descriptor_type == 'ScanContext':
            if descriptor_params is None:
                descriptor_params = {}
            self.descriptor_extractor = ScanContextDescriptor(**descriptor_params)
        else:
            raise ValueError(f"Unknown descriptor type: {descriptor_type}")
        
        # Database storage
        self.descriptors: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []  # Store session, scan_id, pose, etc.
    
    def add_descriptors(
        self,
        descriptors: List[np.ndarray],
        metadata: List[Dict[str, Any]]
    ):
        """
        Add descriptors to the database.
        
        Args:
            descriptors: List of descriptor matrices
            metadata: List of metadata dictionaries (one per descriptor)
        """
        if len(descriptors) != len(metadata):
            raise ValueError("Number of descriptors must match number of metadata entries")
        
        self.descriptors.extend(descriptors)
        self.metadata.extend(metadata)
    
    def compute_similarity(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        handle_rotation: bool = True
    ) -> Tuple[float, Optional[int]]:
        """
        Compute similarity between two descriptors.
        
        Args:
            desc1: First descriptor
            desc2: Second descriptor
            handle_rotation: Whether to handle rotation (circular shift for ScanContext)
            
        Returns:
            Tuple of (similarity_score, best_shift) where best_shift is None if rotation not handled
        """
        if handle_rotation and self.descriptor_type == 'ScanContext':
            similarity, shift = self.descriptor_extractor.match_with_rotation(
                desc1, desc2, self.similarity_metric
            )
            return similarity, shift
        else:
            # No rotation handling
            if self.similarity_metric == 'cosine':
                vec1 = desc1.flatten()
                vec2 = desc2.flatten()
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 > 0 and norm2 > 0:
                    similarity = dot_product / (norm1 * norm2)
                else:
                    similarity = 0.0
                return similarity, None
                
            elif self.similarity_metric == 'euclidean':
                diff = desc1 - desc2
                distance = np.linalg.norm(diff)
                return -distance, None  # Negative distance as similarity
                
            elif self.similarity_metric == 'l1':
                diff = desc1 - desc2
                distance = np.sum(np.abs(diff))
                return -distance, None  # Negative distance as similarity
            else:
                raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
    
    def query_knn(
        self,
        query_descriptor: np.ndarray,
        k: int = 5,
        exclude_temporal_neighbors: bool = True,
        temporal_window: int = 50
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Query k-nearest neighbors.
        
        Args:
            query_descriptor: Query descriptor
            k: Number of neighbors to retrieve
            exclude_temporal_neighbors: Whether to exclude scans from same session within temporal window
            temporal_window: Number of scans to exclude on each side (if exclude_temporal_neighbors)
            
        Returns:
            List of tuples (index, similarity_score, metadata) sorted by similarity (descending)
        """
        if len(self.descriptors) == 0:
            return []
        
        similarities = []
        shifts = []
        
        # Compute similarities with all database descriptors
        for desc in self.descriptors:
            sim, shift = self.compute_similarity(query_descriptor, desc)
            similarities.append(sim)
            shifts.append(shift)
        
        # Get indices sorted by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Filter out temporal neighbors if requested
        results = []
        query_metadata = getattr(self, '_current_query_metadata', {})
        
        for idx in sorted_indices:
            if exclude_temporal_neighbors and 'scan_id' in query_metadata:
                # Check if this is a temporal neighbor
                db_metadata = self.metadata[idx]
                if (db_metadata.get('session') == query_metadata.get('session') and
                    'scan_id' in db_metadata):
                    scan_diff = abs(db_metadata['scan_id'] - query_metadata['scan_id'])
                    if scan_diff <= temporal_window:
                        continue  # Skip temporal neighbor
            
            results.append((
                idx,
                similarities[idx],
                {**self.metadata[idx], 'shift': shifts[idx]}
            ))
            
            if len(results) >= k:
                break
        
        return results
    
    def query_radius(
        self,
        query_descriptor: np.ndarray,
        radius: float,
        exclude_temporal_neighbors: bool = True,
        temporal_window: int = 50
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Query all descriptors within similarity radius.
        
        Args:
            query_descriptor: Query descriptor
            radius: Similarity threshold (minimum similarity to include)
            exclude_temporal_neighbors: Whether to exclude scans from same session within temporal window
            temporal_window: Number of scans to exclude on each side
            
        Returns:
            List of tuples (index, similarity_score, metadata) with similarity >= radius
        """
        if len(self.descriptors) == 0:
            return []
        
        results = []
        query_metadata = getattr(self, '_current_query_metadata', {})
        
        # Compute similarities with all database descriptors
        for idx, desc in enumerate(self.descriptors):
            # Filter temporal neighbors if requested
            if exclude_temporal_neighbors and 'scan_id' in query_metadata:
                db_metadata = self.metadata[idx]
                if (db_metadata.get('session') == query_metadata.get('session') and
                    'scan_id' in db_metadata):
                    scan_diff = abs(db_metadata['scan_id'] - query_metadata['scan_id'])
                    if scan_diff <= temporal_window:
                        continue  # Skip temporal neighbor
            
            sim, shift = self.compute_similarity(query_descriptor, desc)
            
            if sim >= radius:
                results.append((
                    idx,
                    sim,
                    {**self.metadata[idx], 'shift': shift}
                ))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def query(
        self,
        query_descriptor: np.ndarray,
        method: str = 'knn',
        k: int = 5,
        radius: Optional[float] = None,
        query_metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Query the database.
        
        Args:
            query_descriptor: Query descriptor
            method: Retrieval method ('knn' or 'radius_search')
            k: Number of neighbors (for knn)
            radius: Similarity threshold (for radius_search)
            query_metadata: Metadata for the query (used for temporal neighbor exclusion)
            **kwargs: Additional arguments passed to query methods
            
        Returns:
            List of tuples (index, similarity_score, metadata)
        """
        # Store query metadata for temporal neighbor exclusion
        if query_metadata is not None:
            self._current_query_metadata = query_metadata
        
        if method == 'knn':
            return self.query_knn(query_descriptor, k=k, **kwargs)
        elif method == 'radius_search':
            if radius is None:
                raise ValueError("radius must be provided for radius_search method")
            return self.query_radius(query_descriptor, radius=radius, **kwargs)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
    
    def get_database_size(self) -> int:
        """Get the number of descriptors in the database."""
        return len(self.descriptors)
    
    def clear_database(self):
        """Clear all descriptors and metadata from the database."""
        self.descriptors = []
        self.metadata = []
