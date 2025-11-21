export MODEL_DIR="stabilityai/stable-diffusion-2-1"
# It was removed from huggingface, alternative: 
# export MODEL_DIR="Manojb/stable-diffusion-2-1-base"
export OUTPUT_DIR="/data_hdd2/users/yangchen/projects/SIRR/Dereflection-Any-Image/models/model_decoder"
export TORCH_DISTRIBUTED_DEBUG=DETAIL


accelerate launch /data_hdd2/users/yangchen/projects/SIRR/Dereflection-Any-Image/train_decoder.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir="/data_hdd2/users/yangchen/projects/SIRR/DRR_jsonl" \
 --resolution=512 \
 --learning_rate=3e-5 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --dataloader_num_workers=32 \
 --num_train_epochs=40 \
 --validation_steps=3000 \
 --checkpointing_steps=3000 \
 --enable_xformers_memory_efficient_attention \
 --validation_image "/data_hdd2/users/yangchen/projects/SIRR/dataset/nature/natural_I/1-2_43.jpg"\
 --validation_prompt "remove glass reflection" \
 --validation_jsonl "/data_hdd2/users/yangchen/projects/SIRR/dataset/metadata_test_real.jsonl"  \
 --multiple_datasets "metadata_nature_real.jsonl" "metadata_syn.jsonl" "metadata_drr.jsonl" \
 --multiple_datasets_probabilities 0.3 0.3 0.4 \
 --resume_from_pretrained "/data_hdd2/users/yangchen/projects/SIRR/StableDelight/models/model_dual2/checkpoint-25000" \

