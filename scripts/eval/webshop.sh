export CUDA_VISIBLE_DEVICES=0

python -m eval.webshop.run_eval \
    --data_dir data/eval/webshop/ \
    --max_num_examples 500 \
    --save_dir results/webshop \
    --model_name_or_path /net/nfs/mosaic/day/uniagent/train/output \
    --eval_batch_size 8 \
    --formulation lumos_iterative