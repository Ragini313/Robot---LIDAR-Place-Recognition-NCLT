# main.py
import os
import sys
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from utils.data_loader import NCLTDataLoader
from utils.preprocessing import preprocess_point_cloud
from descriptors.scan_context import ScanContextDescriptor
from matching.retrieval import PlaceRetrievalSystem
from evaluation.metrics import evaluate_system
from evaluation.visualization import plot_performance_analysis

def load_config(config_path: str = 'config/config.yaml'):
    """Load configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """Main pipeline for place recognition."""
    print("=== AMR Lab 3: Place Recognition with LiDAR ===")
    
    # Load configuration
    config = load_config()
    print(f"Project: {config['project']['name']}")
    
    # Initialize data loader
    base_data_path = input("Enter path to NCLT dataset (or press Enter for default './data/raw'): ")
    if not base_data_path:
        base_data_path = "./data/raw"
    
    data_loader = NCLTDataLoader(base_data_path)
    
    # Load data for each session
    sessions_data = {}
    for session in config['project']['sessions']:
        try:
            sessions_data[session] = data_loader.load_session_data(
                session, 
                max_scans=config['data']['max_scans_per_session']
            )
            print(f"Successfully loaded {session}")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print(f"Skipping session {session}")
    
    if not sessions_data:
        print("No session data loaded. Exiting.")
        return
    
    print("\nData loading complete!")
    print(f"Loaded {len(sessions_data)} sessions")
    
    # Initialize descriptor extractor
    descriptor_config = config['descriptor']
    descriptor = ScanContextDescriptor(
        num_sectors=descriptor_config['parameters']['num_sectors'],
        num_rings=descriptor_config['parameters']['num_rings'],
        max_range=descriptor_config['parameters']['max_range']
    )
    print(f"\nInitialized {descriptor_config['type']} descriptor")
    
    # Initialize retrieval system
    matching_config = config['matching']
    retrieval_system = PlaceRetrievalSystem(
        descriptor_type=descriptor_config['type'],
        similarity_metric=matching_config['similarity_metric'],
        descriptor_params=descriptor_config['parameters']
    )
    print(f"Initialized retrieval system with {matching_config['similarity_metric']} similarity")
    
    # Extract descriptors and build database
    print("\n=== Extracting Descriptors ===")
    database_descriptors = []
    database_poses = []
    database_metadata = []
    
    downsample_rate = config['data'].get('downsample_rate', 1)
    
    for session_name, session_data in sessions_data.items():
        print(f"\nProcessing session: {session_name}")
        synchronized_scans = session_data.get('synchronized_scans', [])
        
        if not synchronized_scans:
            # Fallback: use velodyne_files directly
            velodyne_files = session_data['velodyne_files']
            gt_data = session_data['ground_truth']
            
            # Simple synchronization: match by index (approximate)
            for scan_idx, vel_file in enumerate(tqdm(velodyne_files[::downsample_rate], desc=f"  Extracting descriptors")):
                try:
                    # Load point cloud
                    point_cloud = data_loader.load_velodyne_scan(vel_file)
                    
                    # Extract descriptor
                    desc = descriptor.compute(point_cloud)
                    database_descriptors.append(desc)
                    
                    # Get approximate pose (use first pose if available)
                    if len(gt_data) > 0:
                        pose_row = gt_data.iloc[min(scan_idx * downsample_rate, len(gt_data) - 1)]
                        pose = {
                            'x': float(pose_row['x']),
                            'y': float(pose_row['y']),
                            'z': float(pose_row['z']),
                            'roll': float(pose_row['roll']) if pd.notna(pose_row['roll']) else 0.0,
                            'pitch': float(pose_row['pitch']) if pd.notna(pose_row['pitch']) else 0.0,
                            'yaw': float(pose_row['yaw']) if pd.notna(pose_row['yaw']) else 0.0
                        }
                    else:
                        pose = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
                    
                    database_poses.append(pose)
                    database_metadata.append({
                        'session': session_name,
                        'scan_id': scan_idx,
                        'filepath': str(vel_file)
                    })
                except Exception as e:
                    print(f"    Error processing {vel_file}: {e}")
                    continue
        else:
            # Use synchronized scans
            for scan_idx, (vel_file, pose_row) in enumerate(tqdm(synchronized_scans[::downsample_rate], desc=f"  Extracting descriptors")):
                try:
                    # Load point cloud
                    point_cloud = data_loader.load_velodyne_scan(vel_file)
                    
                    # Extract descriptor
                    desc = descriptor.compute(point_cloud)
                    database_descriptors.append(desc)
                    
                    # Extract pose (ensure numeric types)
                    pose = {
                        'x': float(pose_row['x']),
                        'y': float(pose_row['y']),
                        'z': float(pose_row['z']),
                        'roll': float(pose_row['roll']) if pd.notna(pose_row['roll']) else 0.0,
                        'pitch': float(pose_row['pitch']) if pd.notna(pose_row['pitch']) else 0.0,
                        'yaw': float(pose_row['yaw']) if pd.notna(pose_row['yaw']) else 0.0
                    }
                    database_poses.append(pose)
                    database_metadata.append({
                        'session': session_name,
                        'scan_id': scan_idx,
                        'filepath': str(vel_file)
                    })
                except Exception as e:
                    print(f"    Error processing {vel_file}: {e}")
                    continue
    
    # Add descriptors to retrieval system
    print(f"\n=== Building Database ===")
    retrieval_system.add_descriptors(database_descriptors, database_metadata)
    print(f"Database size: {retrieval_system.get_database_size()} descriptors")
    
    # Prepare query set
    print(f"\n=== Preparing Query Set ===")
    eval_config = config['evaluation']
    
    # For sequence test: use scans from same sessions, but skip temporal neighbors
    # For cross-session test: use scans from different sessions
    query_descriptors = []
    query_poses = []
    query_metadata = []
    
    if eval_config.get('sequence_test', True):
        # Use every Nth scan as query (skip temporal neighbors)
        query_step = eval_config.get('query_step', 10)  # Query every Nth scan (configurable)
        for i in range(0, len(database_descriptors), query_step):
            query_descriptors.append(database_descriptors[i])
            query_poses.append(database_poses[i])
            query_metadata.append(database_metadata[i])
    else:
        # Use all as queries (for testing)
        query_descriptors = database_descriptors
        query_poses = database_poses
        query_metadata = database_metadata
    
    print(f"Number of queries: {len(query_descriptors)}")
    
    # Run evaluation
    print(f"\n=== Running Evaluation ===")
    evaluation_results = evaluate_system(
        retrieval_system=retrieval_system,
        query_descriptors=query_descriptors,
        query_poses=query_poses,
        query_metadata=query_metadata,
        database_poses=database_poses,
        positive_threshold=eval_config['positive_threshold'],
        negative_threshold=eval_config['negative_threshold'],
        k=matching_config['k_neighbors']
    )
    
    # Print results
    print(f"\n=== Evaluation Results ===")
    overall_metrics = evaluation_results['overall_metrics']
    print(f"Mean Precision: {overall_metrics['mean_precision']:.4f} ± {overall_metrics['std_precision']:.4f}")
    print(f"Mean Recall: {overall_metrics['mean_recall']:.4f} ± {overall_metrics['std_recall']:.4f}")
    print(f"Mean F1 Score: {overall_metrics['mean_f1_score']:.4f} ± {overall_metrics['std_f1_score']:.4f}")
    print(f"Mean Average Precision: {overall_metrics['mean_average_precision']:.4f} ± {overall_metrics['std_average_precision']:.4f}")
    print(f"Number of queries: {overall_metrics['num_queries']}")
    
    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    print(f"\n=== Saving Results ===")
    # Save metrics to JSON
    results_file = results_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            'overall_metrics': overall_metrics,
            'num_queries': len(evaluation_results['per_query_results'])
        }
        json.dump(json_results, f, indent=2)
    print(f"Saved metrics to {results_file}")
    
    # Generate and save plots
    plot_dir = results_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)
    plot_performance_analysis(evaluation_results, save_dir=str(plot_dir), show_plots=False)
    
    print("\nPipeline execution complete!")

if __name__ == "__main__":
    main()
