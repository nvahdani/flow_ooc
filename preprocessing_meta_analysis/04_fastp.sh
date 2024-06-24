#!/usr/bin/env bash
#SBATCH --job-name="fastp on raw data"
#SBATCH --partition=pibu_el8
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8000M
#SBATCH --time=02:00:00
#SBATCH --mail-user=negar.vahdani@students.unibe.ch
#SBATCH --mail-type=begin,end
#SBATCH --output=/data/users/nvahdani/flow_project/meta-analysis/fastp_elife/output_fastp_elife_%j.o
#SBATCH --error=/data/users/nvahdani/flow_project/meta-analysis/fastp_elife/error_fastp_elife_%j.e

module load fastp/0.23.4-GCC-10.3.0;


INPUT_DIR=/data/users/nvahdani/flow_project/meta-analysis/data_elife/raw_fastq
OUTPUT_DIR=/data/users/nvahdani/flow_project/meta-analysis/data_elife/trimmed_fastq

for file in ${INPUT_DIR}/*.fastq; do
    BASENAME=$(basename "$file" .fastq)
    OUTPUT_FILE=${OUTPUT_DIR}/${BASENAME}.trimmed.fastq
    fastp  -i "$file" -o "$OUTPUT_FILE"  --dedup --overrepresentation_analysis  --trim_poly_g
    echo "Adapter removal and trimming completed for $file."
done

## remove the dedup and the trim-poly g