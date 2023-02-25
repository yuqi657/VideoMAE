OUTPUT_DIR='/root/autodl-tmp/code/videomae_wrapper/ckpts/coin_test'
DATA_PATH='/root/autodl-tmp/data/COIN/split_anno'
#MODEL_PATH='/root/autodl-tmp/pretrained_models/videomae/checkpoint_base.pth'
MODEL_PATH='/root/autodl-tmp/code/videomae_wrapper/ckpts/coin_base/checkpoint-8/mp_rank_00_model_states.pt'

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port 12320 \
    run_class_finetuning.py \
    --eval \
    --model vit_base_patch16_224 \
    --data_set COIN \
    --nb_classes 779 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 16 \
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