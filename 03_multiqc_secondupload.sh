#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1000MB
#SBATCH --time=01:00:00
#SBATCH --job-name=multiqc_secondupload
#SBATCH --output=/data/users/nvahdani/flow_project/multiqc_secondupload/output_multiqc_secondupload_%j.o
#SBATCH --error=/data/users/nvahdani/flow_project/multiqc_secondupload/error_multiqc_secondupload_%j.e

module load UHTS/Analysis/MultiQC/1.8
QC_REPORTS_DIR="/data/users/nvahdani/flow_project/fastqc_secondupload/output/*_fastqc.zip"
OUTPUT_DIR="/data/users/nvahdani/flow_project/multiqc_secondupload/output"
multiqc $QC_REPORTS_DIR -o $OUTPUT_DIR

echo "MultiQC report generated in $OUTPUT_DIR"