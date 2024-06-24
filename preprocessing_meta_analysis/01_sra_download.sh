#!/bin/bash
# Job name
#SBATCH --job-name="SRA_download"
# Runtime and memory
#SBATCH --time=02:00:00
# Partition
#SBATCH --partition=pibu_el8
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-6
#SBATCH --output=/data/users/nvahdani/flow_project/meta-analysis/data_elife/error_loaddata_%j.e
#SBATCH --error=/data/users/nvahdani/flow_project/meta-analysis/data_elife/output_loaddata_%j.o
 
# Load SRA Toolkit module
module load SRA-Toolkit/3.0.5-gompi-2021a;
 
 
# Define accession list file
ACCESSION_LIST=/data/users/nvahdani/flow_project/meta-analysis/data_elife/accession_list.txt
 
 
 
# Define output directory
OUTPUT_DIR="/data/users/nvahdani/flow_project/meta-analysis/data_elife/raw_fastq"
 
# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR
 
# Split accession list into arrays
ACCESSIONS=($(cat $ACCESSION_LIST))
 
# Download files from accession list
ACCESSION=${ACCESSIONS[$SLURM_ARRAY_TASK_ID]}
echo "Downloading $ACCESSION"
prefetch $ACCESSION --max-size 40G
fasterq-dump --threads 8 --outdir $OUTPUT_DIR $ACCESSION 
echo "Downloaded $ACCESSION"
