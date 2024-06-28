#!/usr/bin/env bash
#SBATCH --job-name="QC on raw data"
#SBATCH --partition=pibu_el8
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1000M
#SBATCH --time=03:00:00
#SBATCH --job-name=fastqc
#SBATCH --mail-user=negar.vahdani@students.unibe.ch
#SBATCH --mail-type=begin,end
#SBATCH --output=/meta-analysis/fastq_elife/error_fastqc_elife_%j.e
#SBATCH --error=/meta-analysis/fastq_elife/output_fastqc_elife%j.o

module load FastQC/0.11.9-Java-11;
fastqc -o /meta-analysis/fastq_elife -f fastq /meta-analysis/data_elife/raw_fastq/*.fastq



