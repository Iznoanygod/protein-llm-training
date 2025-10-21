module load conda
module use /home/x-apark4/privatemodules
module load conda-env/unsloth-py3.12.8

export GPUS_PER_NODE=2

export LAUNCHER="accelerate launch \
    --num_processes 2 \
    --rdzv_backend c10d \
    --config_file accelerate.yaml \
    --multi_gpu \
    "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER multinode.py" 
$CMD