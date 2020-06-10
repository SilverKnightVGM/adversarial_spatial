#!/bin/bash -l
# NOTE the -l flag!
#

# This is an example job file for a Serial Multi-Process job.
# Note that all of the following statements below that begin
# with #SBATCH are actually commands to the SLURM scheduler.
# Please copy this file to your home directory and modify it
# to suit your needs.
# 
# If you need any help, please email rc-help@rit.edu
#

# Name of the job - You'll probably want to customize this.
#SBATCH -J NOgreeble_keras

# Standard out and Standard Error output files
#SBATCH -o /home/ec1018/greebles_inv/misc_tests/finetuned-resnet50-keras-master/greeble_keras.o
#SBATCH -e /home/ec1018/greebles_inv/misc_tests/finetuned-resnet50-keras-master/greeble_keras.e

#To send emails, set the adcdress below and remove one of the "#" signs.
#SBATCH --mail-user ec1018@rit.edu

# notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-type=ALL

# 5 day run time MAX, anything over will be KILLED
# Request 1 days and 0 hours 
#SBATCH -t 1-0:0:0

# Put the job in the "work" partition and request FOUR cores for one task
# "work" is the default partition so it can be omitted without issue.
#SBATCH -A blr -p tier3 -n 1 -c 4

# Job memory requirements in MB (default), GB=g, or TB=t
# Request 3 GB
#SBATCH --mem=10g

##SBATCH --gres=gpu:v100:1
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:p4:2

#
# Your job script goes below this line.  
#
spack load py-horovod
spack load py-scikit-learn/casknpp
python /home/ec1018/greebles_inv/misc_tests/finetuned-resnet50-keras-master/resnet50_train.py
