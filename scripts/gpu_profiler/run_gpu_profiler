#!/bin/bash
#PBS -l ngpus=1
#PBS -l ncpus=8
#PBS -l walltime=24:00:00
#PBS -q workq@e5-cse-cbgpu01.eecscl.psu.edu
#PBS -N profile_gpu
#PBS -M abs5688@psu.edu
#PBS -m bea
#PBS -l mem=100g
#cd $PBS_O_WORKDIR

###
##  You may have set your $PATH variable depending on your situation -- if your error output can't find commands, you probably need to set your $PATH
#
##  If you have your working environment/shell set up to support running your code, you can try using "qsub -V job-name" to submit your job using all of your current environmental variables.
###

USER=abs5688

export PATH=/scratch/$USER/anaconda3/bin:/scratch/$USER/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/opt/SUNWspro/bin:/usr/ccs/bin:/usr/bin:/usr/sfw/bin:/usr/ucb:/usr/sbin:/usr/lib:/sbin:/usr/dt/bin:/usr/java/bin:/bin:/usr/X11RC/bin:/usr/X11/bin:/opt/pbs/bin

source activate pytorch
python /scratch/abs5688/astra-sim/scripts/gpu_profiler/gpu_profiler.py 0 /scratch/abs5688/astra-sim/scripts/gpu_profiler/results 512
