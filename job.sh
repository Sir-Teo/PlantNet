#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --gres=gpu
#SBATCH --job-name=Teo

module purge

singularity exec --nv \
	    --overlay /scratch/wz1492/overlay-25GB-500K.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /scratch/wz1492/env.sh;"


models=("squeezenet" "resnet50" "mobilenetv2" "efficientnetb0" "inceptionv3")
batch_sizes=(32 64 128)
learning_rates=(0.001 0.005 0.01)
num_epochs=(100)

data_dir="data/train"
num_classes=12
test_size=0.2
val_size=0.2

idx=$SLURM_ARRAY_TASK_ID
model=${models[$((idx/30))]}
batch_size=${batch_sizes[$((idx/10%3))]}
learning_rate=${learning_rates[$((idx/3%3))]}
num_epoch=${num_epochs[$((idx%3))]}

run_name="${model}_bs${batch_size}_lr${learning_rate}_epochs${num_epoch}"


srun python main.py \
    --model "$model" \
    --data_dir "$data_dir" \
    --num_classes "$num_classes" \
    --batch_size "$batch_size" \
    --learning_rate "$learning_rate" \
    --num_epochs "$num_epoch" \
    --test_size "$test_size" \
    --val_size "$val_size"