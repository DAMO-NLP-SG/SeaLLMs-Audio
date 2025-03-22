dataset=scripts/train_examples.json
model_id_or_path=checkpoints/Qwen2.5-7B-Instruct-Audio-base
output_dir=checkpoints/SeaLLMs-Audio

# for single-node training
NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model_type qwen2-audio-7b-instruct \
    --model_id_or_path $model_id_or_path \
    --output_dir $output_dir \
    --dataset $dataset \
    --deepspeed default-zero2 \
    --num_train_epochs 1 \
    --max_length 8196 \
    --sft_type full \
    --per_device_train_batch_size 1 \
    --ddp_backend nccl

# for multi-node training
# NNODES=4 NODE_RANK=$RANK MASTER_ADDR=$MASTER_ADDR NPROC_PER_NODE=8 \
# swift sft \
#     --model_type qwen2-audio-7b-instruct \
#     --model_id_or_path $model_id_or_path \
#     --output_dir $output_dir \
#     --dataset $dataset \
#     --deepspeed default-zero2 \
#     --num_train_epochs 1 \
#     --max_length 8196 \
#     --sft_type full \
#     --per_device_train_batch_size 1 \
#     --ddp_backend nccl
