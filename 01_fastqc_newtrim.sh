#!/usr/bin/env bash

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1000M
#SBATCH --time=03:00:00
#SBATCH --job-name=fastqctrim
#SBATCH --output=/data/users/nvahdani/flow_project/fastqc_newtrim/output_fastqc_newtrim_%j.o
#SBATCH --error=/data/users/nvahdani/flow_project/fastqc_newtrim/error_fastqc_newtrim_%j.e

module add UHTS/Quality_control/fastqc/0.11.9;
fastqc -o /data/users/nvahdani/flow_project/fastqc_newtrim/output -f fastq /data/users/nvahdani/flow_project/reads/trimmed_data_newversion/*.fastq.gz
