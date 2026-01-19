# AMR Lab 3: Place Recognition with LiDAR

## Project Structure


## Dataset Setup

1. Download NCLT dataset sessions (first 3 days):
   - 2012-01-08
   - 2012-01-15  
   - 2012-01-22

2. Download only:
   - Velodyne data (vel.tar.gz)
   - Ground truth pose data (groundtruth.csv)

3. Extract each session to: `data/raw/[session_name]/`

## Installation

```bash
pip install -r requirements.txt


Tasks

    Descriptor Implementation: Implement ScanContext or similar descriptor

    Matching Algorithm: Implement similarity measures and retrieval

    Evaluation: Compute precision, recall, F1, AP, and plot curves
    
    
    
    
## Your Next Steps:

Now you have a well-organized project structure. Your next tasks should be:

1. **Download the dataset** according to the instructions
2. **Extract the data** into the `data/raw/` folder with proper session folders
3. **Implement the descriptor** (start with ScanContext from the provided papers)
4. **Test data loading** with the notebooks

Would you like me to guide you through:
1. Downloading and extracting the NCLT dataset?
2. Implementing the ScanContext descriptor?
3. Or starting with data exploration in the notebook?

Which step would you like to tackle next?
