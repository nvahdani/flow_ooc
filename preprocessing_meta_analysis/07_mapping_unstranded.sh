#!/usr/bin/env bash
#SBATCH --partition=pibu_el8
#SBATCH --array=0-6
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000MB
#SBATCH --time=08:00:00
#SBATCH --job-name=mapping 
#SBATCH --mail-user=negar.vahdani@unibe.ch
#SBATCH --mail-type=end
#SBATCH --error=/data/users/nvahdani/flow_project/meta-analysis/mapping_unstranded/error_mapping_trimmed_%j.e
#SBATCH --output=/data/users/nvahdani/flow_project/meta-analysis/mapping_unstranded/output_mapping_trimmed_%j.o

#load module
module load HISAT2/2.2.1-gompi-2021a ;
#reference genome: Hisat2 indexed reference with transcripts
REF_GENOME="/data/users/nvahdani/flow_project/index110/output/genome_tran"
OUTPUT_DIR="/data/users/nvahdani/flow_project/meta-analysis/mapping_unstranded"
INPUT_DIR=/data/users/nvahdani/flow_project/meta-analysis/data_elife/trimmed_fastq
mkdir -p $OUTPUT_DIR
#creat an array on the raw data
ReadArray=($INPUT_DIR/*.trimmed.fastq)

# Select the file corresponding to the SLURM array task ID
FastqFile=${ReadArray[$SLURM_ARRAY_TASK_ID]}

# Extract the base name without the path and extension
base=$(basename "$FastqFile" .trimmed.fastq)

echo "Processing file: $base"

#run hisat2
hisat2 -p 10 -x $REF_GENOME -U "$FastqFile" -S "$OUTPUT_DIR/${base}_mapping.sam" --new-summary

