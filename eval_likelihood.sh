output_dir="./test_output_eval/salmon/test"

if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi
conf_path="conf/1b_extended.yaml"
data_dir="/share/data/speech/jjery2243542/data/salmon"
id="/share/data/speech/jjery2243542/data/continuous_gslm/salmon/test.txt" # the id file for evaluation, each line is a relative path from data_dir, with the extension defined in conf.yaml
ckpt_path="/share/data/speech/jjery2243542/continuous_gslm/ckpt/MLSEn+people/FM/5e-4/85k/reduction_1/token_loss_weight_1.0/n_res_blocks_6/ll_elm_8bit_fm_mimi_token_conditioning_future_4_cond_future_1b_extended/model-step=0095000.ckpt/modified_ckpt/1b_extend.bin"
k_future_tokens=4
batch_size=1
python trainer.py --data_dir $data_dir --conf $conf_path  --predict_id_file $id --override "{'model': {'flash_attention': False}, 'training':{'batch_size': $batch_size}, 'data': {'ext': 'wav'}}" --ckpt_path $ckpt_path --predict_only --prediction_output_dir $output_dir --use_k_future_tokens $k_future_tokens --ignore_eos 
