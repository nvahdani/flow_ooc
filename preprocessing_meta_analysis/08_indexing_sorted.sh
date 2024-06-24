#!/usr/bin/env bash
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16000MB
#SBATCH --time=01:00:00
#SBATCH --job-name=samtobam
#SBATCH --partition=pibu_el8



# this indexing is for the sorted files 
#saves the indexed bam files with the suffix of .bai in the same directory where sorted bam files exist

#BAMDIR="/data/users/nvahdani/flow_project/sort/output_indexed"
#OUTDIR="/data/users/nvahdani/flow_project/index_sorted/output"
#R1=($(ls -1 /data/users/nvahdani/flow_project/reads/trimmed_data_newversion/*R1*.trimmed.fastq.gz))
#filename=$(basename "${R1[$SLURM_ARRAY_TASK_ID]}"| sed 's/_L.*//')
#base=${filename%.*}

#load module
module load SAMtools/1.13-GCC-10.3.0 ;
samtools index /data/users/nvahdani/flow_project/meta-analysis/bamsort/SRR12654384_sorted.bam