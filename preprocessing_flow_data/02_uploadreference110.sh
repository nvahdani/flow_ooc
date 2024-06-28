#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1000M
#SBATCH --time=03:00:00
#SBATCH --job-name=uploadreference110
#SBATCH --output=/reference110/output_reference_%j.o
#SBATCH --error=/reference110/error_reference_%j.e

# Download the human genome FASTA file and decompress the files
wget -P /reference110 ftp://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
gzip -d /reference110/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
mv /reference110/Homo_sapiens.GRCh38.dna.primary_assembly.fa /data/users/nvahdani/flow_project/reference110/genome.fa
echo FASTA downloaded and unzipped
# Download the human genome GTF file and decompress the files
wget -P /reference110 ftp://ftp.ensembl.org/pub/release-110/gtf/homo_sapiens/Homo_sapiens.GRCh38.110.gtf.gz  
gzip -d /reference110/Homo_sapiens.GRCh38.110.gtf.gz
mv /reference110/Homo_sapiens.GRCh38.110.gtf /data/users/nvahdani/flow_project/reference110/genome.gtf
echo GTF downloaded and unzipped

# Load the module
module load UHTS/Aligner/hisat/2.2.1
# Extract splice sites from the GTF file
hisat2_extract_splice_sites.py /reference110/genome.gtf > /reference110/genome.ss
# Extract exon information from the GTF file
hisat2_extract_exons.py /reference110/genome.gtf > /reference110/genome.exon

echo splice variants and exons generated
