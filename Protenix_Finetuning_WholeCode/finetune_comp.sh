export CUTLASS_PATH="/home/max/Documents/Protenix-KaggleRNA3D/Cutlass/cutlass"
export PROTENIX_DATA_ROOT_DIR="/home/max/Documents/Protenix-KaggleRNA3D/af3-dev/"
export LAYERNORM_TYPE=fast_layernorm
export USE_DEEPSPEED_EVO_ATTENTION=true

python3 runner/train.py \
    --run_name kaggle_rna_finetune \
    --project protenix \
    --base_dir ./output_withmsa \
    --seed 42 \
    --dtype bf16 \
    --use_wandb false \
    --train_confidence_only false \
    --train_crop_size 75 \
    --diffusion_batch_size 4 \
    --iters_to_accumulate 4 \
    --loss.weight.alpha_pae 0.0 \
    --loss.weight.alpha_bond 1.0 \
    --loss.weight.smooth_lddt 0.0 \
    --loss.weight.alpha_diffusion 4.0 \
    --loss.weight.alpha_distogram 0.03 \
    --loss.weight.alpha_confidence 1e-4 \
    --eval_interval 500 \
    --log_interval 50 \
    --checkpoint_interval 1000 \
    --ema_decay 0.999 \
    --max_steps 5 \
    --warmup_steps 1000 \
    --lr 1e-4 \
    --load_checkpoint_path  /home/max/Documents/Protenix-KaggleRNA3D/af3-dev/release_model/model_v0.2.0.pt\
    --load_ema_checkpoint_path /home/max/Documents/Protenix-KaggleRNA3D/af3-dev/release_model/model_v0.2.0.pt \
    --data.train_sets kaggle_rna3d \
    --data.test_sets  kaggle_rna3d \
    --use_wandb false