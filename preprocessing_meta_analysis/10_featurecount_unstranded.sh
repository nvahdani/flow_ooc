#!/usr/bin/env bash
#SBATCH --partition=pibu_el8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --time=02:00:00
#SBATCH --job-name=featurecount
#SBATCH --mail-user=negar.vahdani@unibe.ch
#SBATCH --mail-type=end
#SBATCH --error=/data/users/nvahdani/flow_project/meta-analysis/featurecount_unstranded/error_counttrimmed_%j.e
#SBATCH --output=/data/users/nvahdani/flow_project/meta-analysis/featurecount_unstranded/output_counttrimmed_%j.o



BAMDIR="/data/users/nvahdani/flow_project/meta-analysis/bamsort_unstranded/*.bam"
OUTPUT_DIR="/data/users/nvahdani/flow_project/meta-analysis/featurecount_unstranded"
Annotation_DIR="/data/users/nvahdani/flow_project/reference110"
OUTPUT_FILE="${OUTPUT_DIR}/elife_unstrand_gene_counts.txt"


#load module
module load Subread/2.0.3-GCC-10.3.0  

featureCounts -a $Annotation_DIR/genome.gtf -o $OUTPUT_FILE -s 0 $BAMDIR
