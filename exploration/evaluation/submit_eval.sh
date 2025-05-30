for j in 1; do
    # for i in $(seq 0 48); do
    for i in 000 031; do
        # num=$(printf "%03d" $i)
        num=$i
        python exploration/frederike/evaluation/submit_eval_jobs.py \
        --eval_dir "/fast/fluebeck/biobank/2024_11_29_report/2024_12_10_flexible_model_improved_sampling/evaluation_full/model_$num" \
        --base_prediction_dir "/fast/fluebeck/biobank/2024_11_29_report/2024_12_10_flexible_model_improved_sampling/inference_full/model_$num"
    done
done
