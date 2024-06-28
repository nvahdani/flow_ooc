#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1000M
#SBATCH --time=03:00:00
#SBATCH --job-name=fastqctrim
#SBATCH --output=/fastqc_newtrim/output_fastqc_newtrim_%j.o
#SBATCH --error=/fastqc_newtrim/error_fastqc_newtrim_%j.e

# Load module and run fastqc
module add UHTS/Quality_control/fastqc/0.11.9;
fastqc -o /fastqc_newtrim/output -f fastq /reads/trimmed_data_newversion/*.fastq.gz
