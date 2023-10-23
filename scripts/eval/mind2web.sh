export CUDA_VISIBLE_DEVICES=1

python -m eval.mind2web.run_eval \
    --data_dir data/eval/mind2web/ \
    --max_num_examples 10 \
    --save_dir results/mind2web \
    --model_name_or_path /net/nfs/mosaic/day/uniagent/train/output \
    --eval_batch_size 16 \
    --formulation lumos_iterative