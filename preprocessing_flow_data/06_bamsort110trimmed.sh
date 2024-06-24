#!/usr/bin/env bash
#SBATCH --partition=pall
#SBATCH --nodes=1
#SBATCH --array=0-15
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=25G
#SBATCH --time=02:00:00
#SBATCH --job-name=sorting
#SBATCH --mail-user=negar.vahdani@unibe.ch
#SBATCH --mail-type=end
#SBATCH --error=/bamsort110/error_sorting_%j.e
#SBATCH --output=/bamsort110/output_sorting_%j.o


# define variables
BAMDIR="/samtobam_110/output_trimmed"
OUTDIR="/bamsort110/output_trimmed"
R1=($(ls -1 /reads/trimmed_data_newversion/*R1*.trimmed.fastq.gz))
filename=$(basename "${R1[$SLURM_ARRAY_TASK_ID]}"| sed 's/_L.*//')
base=${filename%.*}

#load module
module load UHTS/Analysis/samtools/1.10

samtools sort -@ 4 -m 24G -o $OUTDIR/${base}_sorted.bam -T temp_bam $BAMDIR/${base}_mapping.bam
