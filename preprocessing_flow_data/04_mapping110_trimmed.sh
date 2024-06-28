#!/usr/bin/env bash
#SBATCH --partition=pall
#SBATCH --nodes=1
#SBATCH --array=0-15
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000MB
#SBATCH --time=08:00:00
#SBATCH --job-name=mapping 
#SBATCH --error=/mapping110/error_mapping110trimmed_%j.e
#SBATCH --output=/mapping110/output_mapping110trimmed_%j.o

# Define the directories
REF_GENOME="/index110/output/genome_tran"
OUTPUT_DIR="/mapping110/output_trimmed"
mkdir -p $OUTPUT_DIR

# Load module
module load UHTS/Aligner/hisat/2.2.1
#reference genome: Hisat2 indexed reference with transcripts

# create an array of the raw data
R1=($(ls -1 /reads/trimmed_data_newversion/*R1*.trimmed.fastq.gz))
filename=$(basename "${R1[$SLURM_ARRAY_TASK_ID]}"| sed 's/_L.*//')
base=${filename%.*}
echo $base

echo -e "${R1[$SLURM_ARRAY_TASK_ID]}"

R2=($(ls -1 /reads/trimmed_data_newversion/*R2*.trimmed.fastq.gz))
echo -e "${R2[$SLURM_ARRAY_TASK_ID]}"
#check the output_DIR

#run hisat2
hisat2 -p 10 -x $REF_GENOME -1 "${R1[$SLURM_ARRAY_TASK_ID]}" -2 "${R2[$SLURM_ARRAY_TASK_ID]}" -S "$OUTPUT_DIR/${base}_mapping.sam" --time --new-summary --rna-strandness FR
