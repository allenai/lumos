export CUDA_VISIBLE_DEVICES=1

python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --max_num_examples 10 \
    --save_dir results/gsm/ \
    --model_name_or_path /net/nfs/mosaic/day/uniagent/train/output \
    --eval_batch_size 16 \
    --formulation lumos_onetime