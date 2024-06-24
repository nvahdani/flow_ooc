#!/usr/bin/env bash

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1000M
#SBATCH --time=03:00:00
#SBATCH --job-name=fastqc_secondupload
#SBATCH --output=/data/users/nvahdani/flow_project/fastqc_secondupload/output_fastqc_secondupload_%j.o
#SBATCH --error=/data/users/nvahdani/flow_project/fastqc_secondupload/error_fastqc_secondupload_%j.e

module add UHTS/Quality_control/fastqc/0.11.9;
fastqc -o /data/users/nvahdani/flow_project/fastqc_secondupload/output -f fastq /data/users/nvahdani/flow_project/reads/data_secondupload/*.fastq.gz
