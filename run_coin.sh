OUTPUT_DIR='YOUR_PATH/ssv2_videomae_pretrain_base_patch16_224_frame_16x2_tube_mask_ratio_0.9_e800/eval_lr_5e-4_epoch_50'
DATA_PATH='YOUR_PATH/list_ssv2'
MODEL_PATH='YOUR_PATH/ssv2_videomae_pretrain_base_patch16_224_frame_16x2_tube_mask_ratio_0.9_e800/checkpoint-799.pth'

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port 12320 --nnodes=1 \
    --node_rank=0 --master_addr=$ip_node_0 \
    run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set COIN \
    --nb_classes 779 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 8 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 64 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 10 \
    --disable_eval_during_finetuning \
    --test_num_segment 2 \
    --test_num_crop 3 \
    --enable_deepspeed