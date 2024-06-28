#!/usr/bin/env bash
#SBATCH --partition=pibu_el8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --time=02:00:00
#SBATCH --job-name=featurecount
#SBATCH --error=/meta-analysis/featurecount_reverse_stranded/error_counttrimmed_%j.e
#SBATCH --output=/meta-analysis/featurecount_reverse_stranded/output_counttrimmed_%j.o


# Define directories
BAMDIR="/meta-analysis/bamsort_unstranded/*.bam"
OUTPUT_DIR="/meta-analysis/featurecount_reverse_stranded"
Annotation_DIR="/reference110"
OUTPUT_FILE="${OUTPUT_DIR}/elife_reverse_strand_gene_counts.txt"


# Load the modules and run featurecount
module load Subread/2.0.3-GCC-10.3.0  

featureCounts -a $Annotation_DIR/genome.gtf -o $OUTPUT_FILE -s 2 $BAMDIR
