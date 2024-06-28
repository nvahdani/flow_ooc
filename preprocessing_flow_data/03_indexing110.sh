#!/usr/bin/env bash
#SBATCH --partition=pall
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16000MB
#SBATCH --time=02:00:00
#SBATCH --job-name=indexing110
#SBATCH --error=/index110/error_indexing_%j.e
#SBATCH --output=/index110/output_indexing_%j.o

# Load module and run
module add UHTS/Aligner/hisat/2.2.1;
hisat2-build -p 16 --exon /reference110/genome.exon --ss /reference110/genome.ss /data/users/nvahdani/flow_project/reference110/genome.fa /index110/genome_tran
