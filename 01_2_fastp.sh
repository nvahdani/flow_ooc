#!/usr/bin/env bash

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1000M
#SBATCH --time=02:00:00
#SBATCH --job-name=fastp
#SBATCH --output=/data/users/nvahdani/flow_project/fastp_newversion/output_fastp_%j.o
#SBATCH --error=/data/users/nvahdani/flow_project/fastp_newversion/error_fastp_%j.e

##module load UHTS/Quality_control/fastp/0.19.5;


INPUT_DIR=/data/users/nvahdani/flow_project/reads/data
OUTPUT_DIR=/data/users/nvahdani/flow_project/reads/trimmed_data_newversion

for R1 in $INPUT_DIR/*_R1_*.fastq.gz
do
    # Define file names for Read 2, trimmed Read 1 and Read 2
    R2="${R1/_R1_*.fastq.gz/_R2_*.fastq.gz}"  # Replace '_R1' with '_R2' in the filename
    TRIMMED_R1="$OUTPUT_DIR/$(basename $R1 .fastq.gz).trimmed.fastq.gz"
    TRIMMED_R2="$OUTPUT_DIR/$(basename $R2 .fastq.gz).trimmed.fastq.gz"
      # Run fastp
    /data/courses/rnaseq_course/tools/fastp -i $R1 -I $R2 -o $TRIMMED_R1 -O $TRIMMED_R2 --detect_adapter_for_pe

    echo "Processed $R1 and $R2"
done

echo "Adapter removal and trimming completed for all files."

