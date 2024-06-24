#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1000MB
#SBATCH --time=01:00:00
#SBATCH --job-name=multiqc
#SBATCH --output=/data/users/nvahdani/flow_project/multiqc_newtrim/output_multiqc_newtrim_%j.o
#SBATCH --error=/data/users/nvahdani/flow_project/multiqc_newtrim/error_multiqc_newtrim_%j.e

module load UHTS/Analysis/MultiQC/1.8
QC_REPORTS_DIR="/data/users/nvahdani/flow_project/fastqc_newtrim/output/*_fastqc.zip"
OUTPUT_DIR="/data/users/nvahdani/flow_project/multiqc_newtrim/output"
multiqc $QC_REPORTS_DIR -o $OUTPUT_DIR

echo "MultiQC report generated in $OUTPUT_DIR"