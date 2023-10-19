export CUDA_VISIBLE_DEVICES=0

python -m eval.hotpotqa.run_eval \
    --data_dir data/eval/hotpotqa/ \
    --max_num_examples 10 \
    --save_dir results/hotpotqa \
    --model_name_or_path /net/nfs/mosaic/day/uniagent/train/output \
    --eval_batch_size 16 \
    --formulation lumos_iterative