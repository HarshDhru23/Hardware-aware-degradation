#!/bin/bash
#SBATCH -N 1                          # Number of nodes
#SBATCH --ntasks-per-node=1           # We are running one python process
#SBATCH --cpus-per-task=8             # Using 8 cores
#SBATCH --time=06:50:20               # Max duration
#SBATCH --job-name=Combine_Histograms # Job name
#SBATCH --error=job.%J.err            # Error file (%J is the JobID)
#SBATCH --output=job.%J.out           # Output file
#SBATCH --partition=small             # Partition

# Initialize Conda for the compute node
# This points to where you installed miniconda3
source ~/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate isro_env

# Set the project directory (adjust if needed)
PROJECT_DIR=/scratch/raj.modi/Hardware-aware-degradation

# Navigate to project directory
cd $PROJECT_DIR

# Find all .npz files in the configs directory
NPZ_FILES=$(find configs/ -name "*.npz" -type f | tr '\n' ' ')

# Check if we found any files
if [ -z "$NPZ_FILES" ]; then
    echo "Error: No .npz files found in configs/ directory"
    exit 1
fi

echo "Found .npz files:"
echo "$NPZ_FILES"
echo ""

# Run the combine_histograms.py script
# It will consume all .npz files and produce combined_stats.yaml and combined_stats.png
python -u combine_histograms.py \
    --histograms $NPZ_FILES \
    --output configs/combined_stats.yaml \
    --plot-output configs/combined_stats.png

echo ""
echo "Job completed!"
echo "Output files:"
echo "  - configs/combined_stats.yaml"
echo "  - configs/combined_stats.png"
