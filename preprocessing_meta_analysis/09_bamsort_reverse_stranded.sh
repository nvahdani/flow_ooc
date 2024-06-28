#!/usr/bin/env bash
#SBATCH --partition=pibu_el8
#SBATCH --nodes=1
#SBATCH --array=0-6
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=25G
#SBATCH --time=02:00:00
#SBATCH --job-name=sorting
#SBATCH --mail-user=negar.vahdani@unibe.ch
#SBATCH --mail-type=end
#SBATCH --error=/meta-analysis/bamsort/error_sorting_%j.e
#SBATCH --output=/meta-analysis/bamsort/output_sorting_%j.o


# define variables
BAMDIR="/meta-analysis/samtobam_reverse_stranded"
OUTDIR="/meta-analysis/bamsort_reverse_stranded"
mkdir -p $OUTDIR

ReadArray=($BAMDIR/*_mapping.bam)

# Select the file corresponding to the SLURM array task ID
FastqFile=${ReadArray[$SLURM_ARRAY_TASK_ID]}

# Extract the base name without the path and extension
base=$(basename "$FastqFile" _mapping.bam)
#load module
module load SAMtools/1.13-GCC-10.3.0 ;

samtools sort -@ 4 -m 24G -o $OUTDIR/${base}_sorted.bam -T temp_bam $BAMDIR/${base}_mapping.bam
