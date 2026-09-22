#!/usr/bin/env bash
set -euo pipefail

STL="phantom_surface.stl"
TRAIN_SEED=42
DEV_SEED=44
EPOCHS=50
TRAIN_SAMPLES=5000
VAL_SAMPLES=500
TARGET_POINTS=1024

if [[ ! -f "$STL" ]]; then
    echo "ERRORE: $STL non trovato."
    exit 1
fi

echo "=== Controllo sintassi e test ==="
python -m py_compile \
    train.py \
    model.py \
    phantom_data.py \
    evaluate.py \
    correspondence_loss.py \
    test_correspondence.py

python -m unittest -v test_correspondence

run_experiment() {
    local mode="$1"
    local weight="$2"
    local source_points="$3"
    local tag="$4"

    local train_exp="ablation_${mode}_corr${tag}_seed${TRAIN_SEED}"
    local eval_exp="dev${DEV_SEED}_${train_exp}"
    local train_log="checkpoints/${train_exp}/train.log"
    local model_path="checkpoints/${train_exp}/models/model.best.t7"
    local eval_log="checkpoints/${eval_exp}/eval.log"

    echo
    echo "======================================================"
    echo "Esperimento: ${train_exp}"
    echo "Mode: ${mode}"
    echo "Source points: ${source_points}"
    echo "corr_weight: ${weight}"
    echo "======================================================"

    if [[ -f "$train_log" ]] &&
       grep -q "Training complete" "$train_log"; then
        echo "Training già completato: ${train_exp}"
    elif [[ -d "checkpoints/${train_exp}" ]]; then
        echo "ERRORE: esiste un training incompleto:"
        echo "checkpoints/${train_exp}"
        echo "Non verrà sovrascritto."
        exit 1
    else
        python train.py \
            --stl "$STL" \
            --exp_name "$train_exp" \
            --emb_nn dgcnn \
            --pointer transformer \
            --head svd \
            --emb_dims 512 \
            --ff_dims 1024 \
            --n_blocks 1 \
            --n_heads 4 \
            --dgcnn_k 20 \
            --batch_size 2 \
            --epochs "$EPOCHS" \
            --scheduler 25 \
            --lr 0.001 \
            --grad_clip 1 \
            --dset_mode "$mode" \
            --dset_num_samples "$TRAIN_SAMPLES" \
            --val_num_samples "$VAL_SAMPLES" \
            --dset_n_points "$source_points" \
            --target_n_points "$TARGET_POINTS" \
            --dset_rot_max 0.17 \
            --dset_trans_max 10 \
            --noise_sigma 0.3 \
            --network_scale_mm 100 \
            --patch_radius_mm 20 \
            --manualSeed "$TRAIN_SEED" \
            --num_workers 4 \
            --corr_weight "$weight" \
            --corr_sigma_mm 5
    fi

    if [[ ! -f "$model_path" ]]; then
        echo "ERRORE: checkpoint migliore non trovato:"
        echo "$model_path"
        exit 1
    fi

    if [[ -f "$eval_log" ]] &&
       grep -q "dcp_tre_mean_mm" "$eval_log"; then
        echo "Valutazione già completata: ${eval_exp}"
    else
        python evaluate.py \
            --model_path "$model_path" \
            --stl "$STL" \
            --num_eval 2000 \
            --batch_size 8 \
            --seed "$DEV_SEED" \
            --exp_name "$eval_exp"
    fi
}

# Sweep: 512 punti nella patch connessa.
run_experiment "sweep" "0.01" "512" "001"
run_experiment "sweep" "0.03" "512" "003"

# Sparse: 25 punti distribuiti sulla superficie.
run_experiment "sparse" "0.01" "25" "001"
run_experiment "sparse" "0.03" "25" "003"

echo
echo "======================================================"
echo "TUTTI GLI ESPERIMENTI SONO TERMINATI"
echo "======================================================"

echo
echo "Riepilogo validation dei training:"
for log in \
    checkpoints/ablation_*_seed42/train.log
do
    echo
    echo "--- $log"
    grep "Training complete. Best val_loss" "$log" || true
done

echo
echo "Riepilogo valutazioni development:"
for log in \
    checkpoints/dev44_ablation_*/eval.log
do
    echo
    echo "--- $log"
    grep -E \
        "Training corr_weight|Mode:|dcp_rotation_deg:|dcp_translation_mm:|dcp_source_proxy_mm:|dcp_tre_mean_mm:|baseline_tre_mean_mm:" \
        "$log" || true
done
