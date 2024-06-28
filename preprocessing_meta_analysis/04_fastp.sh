#!/usr/bin/env bash
#SBATCH --job-name="fastp on raw data"
#SBATCH --partition=pibu_el8
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8000M
#SBATCH --time=02:00:00
#SBATCH --output=/meta-analysis/fastp_elife/output_fastp_elife_%j.o
#SBATCH --error=/meta-analysis/fastp_elife/error_fastp_elife_%j.e

# Define the directories
INPUT_DIR=/meta-analysis/data_elife/raw_fastq
OUTPUT_DIR=/meta-analysis/data_elife/trimmed_fastq

# Load the module
module load fastp/0.23.4-GCC-10.3.0;

# For each file, remove duplicates, overexpressed seq and poly g
for file in ${INPUT_DIR}/*.fastq; do
    BASENAME=$(basename "$file" .fastq)
    OUTPUT_FILE=${OUTPUT_DIR}/${BASENAME}.trimmed.fastq
    fastp  -i "$file" -o "$OUTPUT_FILE"  --dedup --overrepresentation_analysis  --trim_poly_g
    echo "Adapter removal and trimming completed for $file."
done
