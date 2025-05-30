import subprocess

# List of command-line arguments to use with the submit_job script
arguments = [f"{i:03}" for i in range(20)]

base_prediction_dir = (
    "/fast/fluebeck/biobank/2024_08_15_inference_flexible_sampling_longer"
)
eval_dir = "/fast/fluebeck/biobank/2024_08_20_inference_flexible_sampling_results"

# Loop over the arguments and run the submit_job script with each
for arg in arguments:
    subprocess.run(
        [
            "python",
            "exploration/frederike/evaluation/submit_eval_jobs.py",
            "--eval_dir",
            eval_dir + "/" + str(arg),
            "--base_prediction_dir",
            base_prediction_dir + "/" + str(arg),
        ]
    )
    print(base_prediction_dir + "/" + str(arg))
