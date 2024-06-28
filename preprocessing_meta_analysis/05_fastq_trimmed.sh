#!/usr/bin/env bash
# Job name
#SBATCH --job-name="QC on raw data"
#SBATCH --partition=pibu_el8
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1000M
#SBATCH --time=03:00:00
#SBATCH --job-name=fastqc
#SBATCH --mail-type=begin,end
#SBATCH --output=/meta-analysis/fastq_trimmed_elife/error_fastqc_trimmed_elife_%j.e
#SBATCH --error=/meta-analysis/fastq_trimmed_elife/output_fastqc_trimmed_elife%j.o

# Load the module and run fastqc
module load FastQC/0.11.9-Java-11;
fastqc -o /meta-analysis/fastq_trimmed_elife -f fastq /meta-analysis/data_elife/trimmed_fastq/*.fastq



