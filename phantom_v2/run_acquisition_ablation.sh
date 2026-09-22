#!/usr/bin/env bash
set -euo pipefail

STL="phantom_surface.stl"
TRAIN_SEED=42
DEV_SEED=44

run_experiment() {
    local mode="$1"
    local source_points="$2"
    local radius="$3"
    local weight="$4"
    local exp_name="$5"

    local train_log="checkpoints/${exp_name}/train.log"
    local model_path="checkpoints/${exp_name}/models/model.best.t7"
    local eval_name="dev${DEV_SEED}_${exp_name}"
    local eval_log="checkpoints/${eval_name}/eval.log"

    echo
    echo "======================================================"
    echo "$exp_name"
    echo "mode=$mode, points=$source_points, radius=$radius"
    echo "corr_weight=$weight"
    echo "======================================================"

    if [[ -f "$train_log" ]] &&
       grep -q "Training complete" "$train_log"; then
        echo "Training già completato."
    elif [[ -d "checkpoints/${exp_name}" ]]; then
        echo "ERRORE: cartella di training incompleto esistente:"
        echo "checkpoints/${exp_name}"
        exit 1
    else
        python train.py \
            --stl "$STL" \
            --exp_name "$exp_name" \
            --emb_nn dgcnn \
            --pointer transformer \
            --head svd \
            --emb_dims 512 \
            --ff_dims 1024 \
            --n_blocks 1 \
            --n_heads 4 \
            --dgcnn_k 20 \
            --batch_size 2 \
            --epochs 50 \
            --scheduler 25 \
            --lr 0.001 \
            --grad_clip 1 \
            --dset_mode "$mode" \
            --dset_num_samples 5000 \
            --val_num_samples 500 \
            --dset_n_points "$source_points" \
            --target_n_points 1024 \
            --dset_rot_max 0.17 \
            --dset_trans_max 10 \
            --noise_sigma 0.3 \
            --network_scale_mm 100 \
            --patch_radius_mm "$radius" \
            --manualSeed "$TRAIN_SEED" \
            --num_workers 4 \
            --corr_weight "$weight" \
            --corr_sigma_mm 5
    fi

    if [[ ! -f "$model_path" ]]; then
        echo "ERRORE: checkpoint non trovato: $model_path"
        exit 1
    fi

    if [[ -f "$eval_log" ]] &&
       grep -q "dcp_tre_mean_mm" "$eval_log"; then
        echo "Valutazione già completata."
    else
        python evaluate.py \
            --model_path "$model_path" \
            --stl "$STL" \
            --num_eval 2000 \
            --batch_size 8 \
            --seed "$DEV_SEED" \
            --exp_name "$eval_name"
    fi
}

# Sweep: radius 20 è già disponibile.
# Nuovi radius 40 e 60, standard e supervisionato.
run_experiment sweep 512 40 0.0 \
    "acq_sweep_r40_dcp_seed42"

run_experiment sweep 512 40 0.03 \
    "acq_sweep_r40_corr003_seed42"

run_experiment sweep 512 60 0.0 \
    "acq_sweep_r60_dcp_seed42"

run_experiment sweep 512 60 0.03 \
    "acq_sweep_r60_corr003_seed42"

# Sparse: 25 punti sono già disponibili.
# Nuovi valori 20, 35 e 50, standard e supervisionato.
run_experiment sparse 20 20 0.0 \
    "acq_sparse_n20_dcp_seed42"

run_experiment sparse 20 20 0.10 \
    "acq_sparse_n20_corr010_seed42"

run_experiment sparse 35 20 0.0 \
    "acq_sparse_n35_dcp_seed42"

run_experiment sparse 35 20 0.10 \
    "acq_sparse_n35_corr010_seed42"

run_experiment sparse 50 20 0.0 \
    "acq_sparse_n50_dcp_seed42"

run_experiment sparse 50 20 0.10 \
    "acq_sparse_n50_corr010_seed42"

echo
echo "======================================================"
echo "ABLATION ACQUISIZIONE COMPLETATA"
echo "======================================================"

echo
echo "SWEEP:"

for log in \
    checkpoints/dev44_sweep_corr000_seed42/eval.log \
    checkpoints/dev44_ablation_sweep_corr003_seed42/eval.log \
    checkpoints/dev44_acq_sweep_r40_dcp_seed42/eval.log \
    checkpoints/dev44_acq_sweep_r40_corr003_seed42/eval.log \
    checkpoints/dev44_acq_sweep_r60_dcp_seed42/eval.log \
    checkpoints/dev44_acq_sweep_r60_corr003_seed42/eval.log
do
    echo
    echo "--- $log"
    grep -E \
        "Training corr_weight|Raggio patch:|dcp_rotation_deg:|dcp_translation_mm:|dcp_source_proxy_mm:|dcp_tre_mean_mm:" \
        "$log"
done

echo
echo "SPARSE:"

for log in \
    checkpoints/dev44_sparse_corr000_seed42/eval.log \
    checkpoints/dev44_sparse_corr010_seed42/eval.log \
    checkpoints/dev44_acq_sparse_n20_dcp_seed42/eval.log \
    checkpoints/dev44_acq_sparse_n20_corr010_seed42/eval.log \
    checkpoints/dev44_acq_sparse_n35_dcp_seed42/eval.log \
    checkpoints/dev44_acq_sparse_n35_corr010_seed42/eval.log \
    checkpoints/dev44_acq_sparse_n50_dcp_seed42/eval.log \
    checkpoints/dev44_acq_sparse_n50_corr010_seed42/eval.log
do
    echo
    echo "--- $log"
    grep -E \
        "Training corr_weight|Mode:|dcp_rotation_deg:|dcp_translation_mm:|dcp_source_proxy_mm:|dcp_tcp_tre_mean_mm:" \
        "$log"
done
