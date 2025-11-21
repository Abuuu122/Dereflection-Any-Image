export MODEL_DIR="stabilityai/stable-diffusion-2-1"
# It was removed from huggingface, alternative: 
# export MODEL_DIR="Manojb/stable-diffusion-2-1-base"
export OUTPUT_DIR="/data_hdd2/users/yangchen/projects/SIRR/Dereflection-Any-Image/models/model_dual"
export TORCH_DISTRIBUTED_DEBUG=DETAIL


accelerate launch /data_hdd2/users/yangchen/projects/SIRR/Dereflection-Any-Image/train_dual.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir="/data_hdd2/users/yangchen/projects/SIRR/DRR_jsonl" \
 --resolution=768 \
 --learning_rate=1e-5 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=1 \
 --num_train_epochs=10 \
 --validation_steps=5000 \
 --checkpointing_steps=5000 \
 --enable_xformers_memory_efficient_attention \
 --validation_image "/data_hdd2/users/yangchen/projects/SIRR/dataset/nature/natural_I/1-2_43.jpg"\
 --validation_prompt "remove glass reflection" \
 --validation_jsonl "/data_hdd2/users/yangchen/projects/SIRR/DRR_jsonl/metadata_test_real.jsonl" \
 --multiple_datasets "metadata_syn_dual.jsonl" "metadata_drr_dual.jsonl" \
 --multiple_datasets_probabilities 0.5 0.5 \
 --resume_from_pretrained "/data_hdd2/users/yangchen/projects/SIRR/StableDelight/models/model_dual2/checkpoint-25000" \
