#!/usr/bin/env bash
#SBATCH --partition=pibu_el8
#SBATCH --array=0-6
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000MB
#SBATCH --time=01:00:00
#SBATCH --job-name=samtobam
#SBATCH --mail-user=negar.vahdani@unibe.ch
#SBATCH --mail-type=begin,end
#SBATCH --output=/meta-analysis/samtobam_unstranded/output_samtobam_trimmed%j.o
#SBATCH --error=/meta-analysis/samtobam_unstranded/error_samtobam_trimmed%j.e

OUTDIR="/meta-analysis/samtobam_unstranded"
SAMDIR="/meta-analysis/mapping_unstranded"
mkdir -p $OUTDIR

ReadArray=($SAMDIR/*_mapping.sam)

# Select the file corresponding to the SLURM array task ID
FastqFile=${ReadArray[$SLURM_ARRAY_TASK_ID]}

# Extract the base name without the path and extension
base=$(basename "$FastqFile" _mapping.sam)

echo "Processing file: $base";
module load SAMtools/1.13-GCC-10.3.0 ;
samtools view -hbS $SAMDIR/${base}_mapping.sam  -o $OUTDIR/${base}_mapping.bam
