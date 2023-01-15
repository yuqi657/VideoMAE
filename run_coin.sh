OUTPUT_DIR='/root/autodl-tmp/code/videomae_wrapper/ckpts/coin_base'
DATA_PATH='/root/autodl-tmp/data/COIN/split_anno'
MODEL_PATH='/root/autodl-tmp/pretrained_models/videomae/checkpoint_base.pth'

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port 12320 \
    run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set COIN \
    --nb_classes 779 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 2 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 1 \
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