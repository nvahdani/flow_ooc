#!/usr/bin/env bash

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1000M
#SBATCH --time=03:00:00
#SBATCH --job-name=uploadreference110
#SBATCH --mail-user=negar.vahdani@unibe.ch
#SBATCH --mail-type=end
#SBATCH --output=/data/users/nvahdani/flow_project/reference110/output_reference_%j.o
#SBATCH --error=/data/users/nvahdani/flow_project/reference110/error_reference_%j.e

##FASTA
wget -P /data/users/nvahdani/flow_project/reference110 ftp://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
gzip -d /data/users/nvahdani/flow_project/reference110/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
mv /data/users/nvahdani/flow_project/reference110/Homo_sapiens.GRCh38.dna.primary_assembly.fa /data/users/nvahdani/flow_project/reference110/genome.fa
echo FASTA downloaded and unzipped
##GTF
wget -P /data/users/nvahdani/flow_project/reference110 ftp://ftp.ensembl.org/pub/release-110/gtf/homo_sapiens/Homo_sapiens.GRCh38.110.gtf.gz  
gzip -d /data/users/nvahdani/flow_project/reference110/Homo_sapiens.GRCh38.110.gtf.gz
mv /data/users/nvahdani/flow_project/reference110/Homo_sapiens.GRCh38.110.gtf /data/users/nvahdani/flow_project/reference110/genome.gtf
echo GTF downloaded and unzipped


module load UHTS/Aligner/hisat/2.2.1

hisat2_extract_splice_sites.py /data/users/nvahdani/flow_project/reference110/genome.gtf > /data/users/nvahdani/flow_project/reference110/genome.ss
hisat2_extract_exons.py /data/users/nvahdani/flow_project/reference110/genome.gtf > /data/users/nvahdani/flow_project/reference110/genome.exon

echo splice variants and exons generated
