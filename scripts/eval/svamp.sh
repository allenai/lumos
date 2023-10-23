export CUDA_VISIBLE_DEVICES=0,1

python -m eval.svamp.run_eval \
    --data_dir data/eval/svamp/ \
    --max_num_examples 10 \
    --save_dir results/svamp/ \
    --model_name_or_path /net/nfs/mosaic/day/uniagent/train/output \
    --eval_batch_size 32 \
    --formulation lumos_onetime