#!/usr/bin/env bash
#SBATCH --partition=pall
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --time=02:00:00
#SBATCH --job-name=featurecount
#SBATCH --mail-user=negar.vahdani@unibe.ch
#SBATCH --mail-type=end
#SBATCH --error=/featurecount110/error_counttrimmed_%j.e
#SBATCH --output=/featurecount110/output_counttrimmed_%j.o



BAMDIR="/bamsort110/output_trimmed/*.bam"
OUTPUT_DIR="/featurecount110/output_trimmed"
Annotation_DIR="/reference110"
OUTPUT_FILE="${OUTPUT_DIR}/gene_counts.txt"


#load module
module load UHTS/Analysis/subread/2.0.1;

featureCounts -p -a $Annotation_DIR/genome.gtf -o $OUTPUT_FILE -s 1 $BAMDIR
