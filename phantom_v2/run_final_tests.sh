#!/usr/bin/env bash
# Test finali sui modelli già addestrati (nessun nuovo training):
#   1) test held-out (seed mai usato nelle scelte progettuali);
#   2) bacino di convergenza: disallineamento iniziale crescente;
#   3) robustezza al rumore del tracker.
# Ogni valutazione riporta TRE, success rate (TRE < 5 mm)
# e salva i casi peggiori per l'analisi dei fallimenti.
set -euo pipefail

STL="phantom_surface.stl"
TEST_SEED=314159
SUCCESS_TRE_MM=5

# Configurazioni scelte sul development set (seed 44).
SWEEP_POSE="acq_sweep_r60_dcp_seed42"
SWEEP_KL="acq_sweep_r60_corr003_seed42"
SPARSE_POSE="acq_sparse_n50_dcp_seed42"
SPARSE_KL="acq_sparse_n50_corr010_seed42"

run_eval() {
    local exp="$1"
    local eval_name="$2"
    local num_eval="$3"
    shift 3

    local model_path="checkpoints/${exp}/models/model.best.t7"
    local eval_log="checkpoints/${eval_name}/eval.log"

    if [[ ! -f "$model_path" ]]; then
        echo "ERRORE: checkpoint non trovato: $model_path"
        exit 1
    fi

    if [[ -f "$eval_log" ]] && grep -q "dcp_success_rate" "$eval_log"; then
        echo "Già valutato: $eval_name"
        return
    fi

    echo
    echo "=== $eval_name"
    python evaluate.py \
        --model_path "$model_path" \
        --stl "$STL" \
        --num_eval "$num_eval" \
        --batch_size 8 \
        --seed "$TEST_SEED" \
        --success_tre_mm "$SUCCESS_TRE_MM" \
        --exp_name "$eval_name" \
        "$@"
}

if [[ ! -f "$STL" ]]; then
    echo "ERRORE: $STL non trovato."
    exit 1
fi

# 1) Test held-out: tabella finale {sweep, sparse} x {DCP, baseline}.
for exp in "$SWEEP_POSE" "$SWEEP_KL" "$SPARSE_POSE" "$SPARSE_KL"; do
    run_eval "$exp" "test_${exp}" 2000
done

# 2) Bacino di convergenza (modelli addestrati a ±0.17 rad, ±10 mm).
#    Livelli: ±10°, ±20°, ±30°, ±45° per asse con ±10, 20, 30, 50 mm.
LEVELS=("0.17 10" "0.35 20" "0.52 30" "0.79 50")

for exp in "$SWEEP_KL" "$SPARSE_KL"; do
    for level in "${LEVELS[@]}"; do
        read -r rot trans <<< "$level"
        run_eval "$exp" "basin_${exp}_rot${rot}_trans${trans}" 1000 \
            --rot_max "$rot" --trans_max "$trans"
    done
done

# 3) Robustezza al rumore (sigma = 0.3 coincide con il test held-out).
for exp in "$SWEEP_KL" "$SPARSE_KL"; do
    for sigma in 0.0 0.6; do
        run_eval "$exp" "noise_${exp}_sigma${sigma}" 1000 \
            --noise_sigma "$sigma"
    done
done

echo
echo "======================================================"
echo "TEST FINALI COMPLETATI"
echo "======================================================"
python summarize_final_tests.py
