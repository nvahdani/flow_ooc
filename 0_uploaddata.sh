#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1000MB
#SBATCH --time=01:00:00
#SBATCH --job-name=data_upload
#SBATCH --output=/data/users/nvahdani/flow_project/reads/data_secondupload/output_secondupload_%j.o
#SBATCH --error=/data/users/nvahdani/flow_project/reads/data_secondupload/error_secondupload_%j.e

wget -i /data/users/nvahdani/flow_project/reads/urls.txt -P /data/users/nvahdani/flow_project/reads/data_secondupload 
