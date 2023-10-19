export CUDA_VISIBLE_DEVICES=1

python -m eval.strategyqa.run_eval \
    --data_dir data/eval/strategyqa/ \
    --max_num_examples 10 \
    --save_dir results/strategyqa \
    --model_name_or_path /net/nfs/mosaic/day/uniagent/train/output \
    --eval_batch_size 32 \
    --formulation lumos_onetime