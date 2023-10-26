export CUDA_VISIBLE_DEVICES=0,1

TASK=complex_qa
MODEL_SIZE=7b
NUM_GPUS=2
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    model/finetune.py \
    --model_name_or_path /net/nfs/mosaic/day/llama_hf/llama-2-${MODEL_SIZE} \
    --use_flash_attn \
    --tokenizer_name /net/nfs/mosaic/day/llama_hf/llama-2-${MODEL_SIZE} \
    --use_slow_tokenizer \
    --train_file data/train/${TASK}/train_annots/lumos_${TASK}_ground_onetime.jsonl \
    --max_seq_length 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --output_dir /net/nfs/mosaic/day/uniagent/train/output/lumos_${TASK}_ground_onetime/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1
