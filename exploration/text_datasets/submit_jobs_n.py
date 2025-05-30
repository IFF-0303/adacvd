import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        required=False,
        help="Base directory for creating the runs",
    )
    parser.add_argument(
        "--low",
        type=int,
    )
    parser.add_argument(
        "--n",
        type=int,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    base_dir = Path(args.base_dir)

    low = args.low
    n = args.n

    for i in range(low, low + n):
        folder_name = str(i).zfill(3)
        os.system(
            f"condor_submit_bid 100 " f"{base_dir/folder_name}/submission_file.sub"
        )
