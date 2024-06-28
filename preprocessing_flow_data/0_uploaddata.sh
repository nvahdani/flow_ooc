#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1000MB
#SBATCH --time=01:00:00
#SBATCH --job-name=data_upload
#SBATCH --output=/reads/data_secondupload/output_secondupload_%j.o
#SBATCH --error=/reads/data_secondupload/error_secondupload_%j.e

# Download the data from the NGS facility website
wget -i /reads/urls.txt -P /reads/data_secondupload 
