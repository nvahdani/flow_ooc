#!/usr/bin/env bash
#SBATCH --job-name="MultiQC on trimmed data"
#SBATCH --partition=pibu_el8
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1000MB
#SBATCH --time=01:00:00
#SBATCH --job-name=multiqc
#SBATCH --output=/meta-analysis/multiqc_trimmed_elife/output_multiqc_elife_%j.o
#SBATCH --error=/meta-analysis/multiqc_trimmed_elife/error_multiqc_elife_%j.e

# Define directories
QC_REPORTS_DIR="/meta-analysis/fastq_trimmed_elife/*_fastqc.zip"
OUTPUT_DIR="/meta-analysis/multiqc_trimmed_elife"
mkdir -p $OUTPUT_DIR

# Load the module
module load MultiQC/1.11-foss-2021a;
multiqc $QC_REPORTS_DIR -o $OUTPUT_DIR

echo "MultiQC report generated in $OUTPUT_DIR"

