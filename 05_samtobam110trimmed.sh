#!/usr/bin/env bash
#SBATCH --array=0-15
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000MB
#SBATCH --time=01:00:00
#SBATCH --job-name=samtobam
#SBATCH --output=/samtobam_110/output_samtobam110trimmed%j.o
#SBATCH --error=/samtobam_110/error_samtobam110trimmed%j.e

OUTDIR="/samtobam_110/output_trimmed"
SAMDIR="/mapping110/output_trimmed"


R1=($(ls -1 /reads/trimmed_data_newversion/*R1*.trimmed.fastq.gz))
filename=$(basename "${R1[$SLURM_ARRAY_TASK_ID]}"| sed 's/_L.*//')
base=${filename%.*}
echo $base
module load UHTS/Analysis/samtools/1.10

samtools view -hbS $SAMDIR/${base}_mapping.sam  -o $OUTDIR/${base}_mapping.bam
