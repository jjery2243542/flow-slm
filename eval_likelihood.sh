#!/usr/bin/bash
#SBATCH --partition=speech-gpu
##SBATCH --partition=cpu
##SBATCH --gpus=nvidia_titan_rtx|nvidia_rtx_a4000:4
#SBATCH -G1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
##SBATCH -C 24g
##SBATCH -C nvidia_titan_rtx|nvidia_rtx_a4000
#SBATCH -o slurm/eval/slurm-%x.out
#SBATCH -e slurm/eval/slurm-%x.err
#SBATCH --open-mode=append
###SBATCH --exclude=gpu3
##SBATCH --signal=SIGHUP@180

source ~/.bashrc
source ~/env/activate_conda
#eval "$(conda shell.bash hook)"
conda activate lm_new_tf
echo $CUDA_VISIBLE_DEVICES

output_dir="./test_output_eval/swuggy/test_reduction"

if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi
conf_path="/share/data/speech-lang/jjery2243542/continuous_gslm/ckpt/reduction_4/270m_reduction/conf.yaml"
#data_dir="/share/data/speech/jjery2243542/data/salmon"
data_dir="/home-nfs/jjery2243542/zr-data/datasets/sLM21-dataset/lexical/dev"
id="/share/data/speech/jjery2243542/data/continuous_gslm/zerospeech/lexical/dev.txt"
ckpt_path="/share/data/speech-lang/jjery2243542/continuous_gslm/ckpt/reduction_4/270m_reduction/last.ckpt/pytorch_model.bin"
#k_future_tokens=4
batch_size=1
python trainer.py --data_dir $data_dir --conf $conf_path  --predict_id_file $id --override "{'model': {'flash_attention': False}, 'training':{'batch_size': $batch_size}, 'data': {'ext': 'wav'}}" --ckpt_path $ckpt_path --predict_only --prediction_output_dir $output_dir --ignore_eos 
