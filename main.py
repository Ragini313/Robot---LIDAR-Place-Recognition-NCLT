# main.py
import os
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from utils.data_loader import NCLTDataLoader
from utils.preprocessing import preprocess_point_cloud
from descriptors.scan_context import ScanContextDescriptor
from matching.retrieval import PlaceRetrievalSystem
from evaluation.metrics import evaluate_system

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
    
    # TODO: Continue with descriptor extraction, matching, and evaluation
    
    print("\nPipeline execution complete!")

if __name__ == "__main__":
    main()
