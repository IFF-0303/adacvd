# this script is used to check the status of training jobs and resubmit them if they have failed.
# it checks the info.err and info.log files for specific error messages and modifies the submission file accordingly.
# it also (optionally) modifies the train_settings.yaml file to reduce the batch size if a CUDA out of memory error is detected.

import os
import re
from os.path import join

import yaml

base_dir = "/fast/groups/hfm-users/pandora-med-box/results/2025_03_10_full_model_inference/2025_03_06_mistral_fgs/model_018/"


def delete_info_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith("info."):
                os.remove(os.path.join(root, file))


def modify_submission_file(submission_file):
    with open(submission_file, "r") as f:
        content = f.read()
    # content = content.replace(
    #     "request_gpus = 8",
    #     "request_gpus = 4",
    # )

    # content = content.replace(
    #     "81000",
    #     "51000",
    # )
    o = 'requirements = (TARGET.CUDAGlobalMemoryMb > 51000) && (Machine != "g099.internal.cluster.is.localnet")'
    n = 'requirements = (TARGET.CUDAGlobalMemoryMb > 81000) && (Machine != "g125.internal.cluster.is.localnet")'
    content = content.replace(o, n)

    # o = "requirements = (TARGET.CUDAGlobalMemoryMb > 81000)"
    # n = 'requirements = (TARGET.CUDAGlobalMemoryMb > 81000) && (Machine != "g125.internal.cluster.is.localnet")'
    content = content.replace(o, n)

    with open(submission_file, "w") as f:
        f.write(content)

    print(f"Modified submission file.")


def modify_train_settings_file(train_settings_file):
    with open(train_settings_file, "r") as f:
        content = f.read()
    content = content.replace("eval_batch_size: 16", "eval_batch_size: 4")

    with open(train_settings_file, "w") as f:
        f.write(content)
    print(f"Modified train_settings file.")


# Function to submit bids
def submit_bid(bid, base_dir, submission_file):
    os.system(f"condor_submit_bid {bid} {join(base_dir, submission_file)}")


def check_info_err(directory):
    err_file_path = os.path.join(base_dir, directory, "info.err")
    if os.path.isfile(err_file_path):
        with open(err_file_path, "r") as f:
            lines = f.readlines()
            node_number = "unknown"
            for line in lines:
                match = re.search(r"INFO: Node: (\S+)", line)

                if match:
                    node_number = match.group(1)

            if lines:
                last_line = lines[-1].strip()
                if last_line.endswith("returned non-zero exit status 1."):
                    print(
                        f"Directory {directory} on node {node_number}: Last line in info.err indicates non-zero exit status."
                    )
            else:
                print(f"Directory {directory}: info.err file is empty.")
    else:
        print(f"Directory {directory}: info.err file not found.")


def check_info_log(directory):
    err_file_path = os.path.join(base_dir, directory, "info.log")
    job_failed = True  # TODO
    if os.path.isfile(err_file_path):
        with open(err_file_path, "r") as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-2].strip()
                if last_line.endswith("with exit-code 1."):
                    print(f"Directory {directory}: Exit-code 1.")
                    # job_failed = True
                elif last_line.endswith("with exit-code 0."):
                    print(f"Directory {directory}: Exit-code 0.")
                    job_failed = False
                else:
                    print(f"Directory {directory}: {last_line}")
            else:
                print(f"Directory {directory}: info.log file is empty.")
    else:
        print(f"Directory {directory}: info.log file not found.")
    return job_failed


def check_text_in_file(file_path, target_text):
    try:
        with open(file_path, "r") as file:
            for line in file:
                if target_text in line:
                    return True
        return False
    except FileNotFoundError:
        print("File not found.")
        return False


# Function to find directories with three-digit numbers
def find_three_digit_directories(base_dir):
    three_digit_directories = []
    if os.path.isdir(base_dir):
        for dir_name in os.listdir(base_dir):
            if re.match(r"\d{3}", dir_name) and os.path.isdir(
                os.path.join(base_dir, dir_name)
            ):
                three_digit_directories.append(dir_name)
    else:
        print(f"Base directory '{base_dir}' not found.")
    three_digit_directories.sort(key=lambda x: int(x))
    return three_digit_directories


import os
import shutil


def move_files_except_specific(source_dir, destination_dir):
    # Create the archive directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # List all files in the source directory
    files = os.listdir(source_dir)

    # Move files to the archive directory
    for file in files:
        # Exclude train_settings.yaml and submission_file.sub
        if (
            file != "train_settings.yaml"
            and file != "submission_file.sub"
            and file != "failed_run"
        ):
            source_path = os.path.join(source_dir, file)
            destination_path = os.path.join(destination_dir, file)
            shutil.move(source_path, destination_path)
            print(f"Moved {file} to {destination_dir}")


def update_train_settings(directory):

    # Open train_settings.yaml and load it as a dictionary
    with open(os.path.join(directory, "train_settings.yaml"), "r") as file:
        d = yaml.safe_load(file)

    # Set the batch_size to 2
    d["training"]["batch_size"] = 2
    d["training"]["eval_steps"] = 2000

    # save the updated dictionary to train_settings.yaml
    with open(os.path.join(directory, "train_settings.yaml"), "w") as file:
        yaml.dump(d, file)


def main():
    # Iterate through directories
    for directory in find_three_digit_directories(base_dir):
        job_failed = check_info_log(directory)
        if job_failed:
            dir_path = os.path.join(base_dir, directory)
            if not os.path.isfile(os.path.join(dir_path, "info.err")):
                continue
            with open(os.path.join(dir_path, "info.err"), "r") as f:
                lines = f.readlines()
                node_number = "unknown"
                for line in lines:
                    match = re.search(r"INFO: Node: (\S+)", line)

                    if match:
                        node_number = match.group(1)
                print(f"Node number: {node_number}")
            rerun = False
            if check_text_in_file(
                os.path.join(dir_path, "info.err"),
                "DistBackendError",
            ):
                print("DistBackendError")
                # with open(os.path.join(dir_path, "inference_settings.yaml"), "r") as f:
                #     config = yaml.safe_load(f)
                # print(config["inference"]["eval_batch_size"])

                # modify_train_settings_file(
                #     os.path.join(dir_path, "inference_settings.yaml")
                # )
                modify_submission_file(os.path.join(dir_path, "submission_file.sub"))

                # modify_submission_file(os.path.join(dir_path, "submission_file.sub"))
                rerun = True
            # elif check_text_in_file(
            #     os.path.join(dir_path, "info.err"),
            #     "CUDA out of memory.",
            # ):
            #     print("CUDA out of memory.")
            #     with open(os.path.join(dir_path, "text_train_settings.yaml"), "r") as f:
            #         config = yaml.safe_load(f)
            #     print(config["training"]["batch_size"])
            #     modify_train_settings_file(
            #         os.path.join(dir_path, "train_settings.yaml")
            #     )
            #     # modify_submission_file(os.path.join(dir_path, "submission_file.sub"))
            #     rerun = True
            # elif check_text_in_file(
            #     os.path.join(dir_path, "info.err"),
            #     "No such file or directory: '/home/fluebeck/.cache/huggingface/datasets",
            # ):
            #     print("Huggingface Cache Error")
            #     # rerun = True
            # elif check_text_in_file(
            #     os.path.join(dir_path, "info.err"),
            #     "failed (exitcode: -7) local_rank: 0 (pid:",
            # ):
            #     print("Local rank error.")
            # elif check_text_in_file(
            #     os.path.join(dir_path, "info.err"),
            #     "Argument list too long",
            # ):
            #     print("Argument list too long.")
            # elif check_text_in_file(
            #     os.path.join(dir_path, "info.err"),
            #     "RuntimeError: [1] is setting up NCCL communicator",
            # ):
            #     print("RuntimeError: [1] is setting up NCCL communicator")
            #     # rerun = True
            # elif check_text_in_file(
            #     os.path.join(dir_path, "info.err"),
            #     "torch.distributed.elastic.multiprocessing.errors.ChildFailedError",
            # ):
            #     print("ChildFailedError")
            # modify_submission_file(os.path.join(dir_path, "submission_file.sub"))
            # modify_train_settings_file(
            #     os.path.join(dir_path, "train_settings.yaml")
            # )
            # rerun = True
            # elif check_text_in_file(
            #     os.path.join(dir_path, "info.log"),
            #     "via condor_rm (by user fluebeck)",
            # ):
            #     print("condor_rm")
            # modify_submission_file(os.path.join(dir_path, "submission_file.sub"))
            # modify_train_settings_file(
            #     os.path.join(dir_path, "train_settings.yaml")
            # )
            # rerun = True
            # elif check_text_in_file(
            #     os.path.join(dir_path, "info.err"),
            #     "Cannot access gated repo",
            # ):
            #     print("Cannot access gated repo")
            # rerun = True

            else:
                print("Unknown error.")

            if rerun:

                print("rerun")
                # move_files_except_specific(
                #     dir_path, os.path.join(dir_path, "failed_run")
                # )
                # update_train_settings(dir_path)

                bid = 100
                submission_file = "submission_file.sub"
                # submission_file_path = os.path.join(dir_path, submission_file)

                # with open(submission_file_path, "r") as file:
                #     for line in file:
                #         match = re.search(r"--main_process_port (\d+)", line)
                #         if match:
                #             port_number = match.group(1)
                #             print("Main process port number:", port_number)
                # modify_submission_file(submission_file_path)
                submit_bid(bid, dir_path, submission_file)
                # break


if __name__ == "__main__":
    main()
