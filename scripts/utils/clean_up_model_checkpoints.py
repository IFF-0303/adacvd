import argparse
import os
import re
import shutil


def cleanup_checkpoint_files(
    base_directory=".", delete_all=False, keep_only_last_epoch=False
):
    """
    Finds and deletes checkpoint files, keeping only the file with the highest number.

    :param base_directory: Directory to start searching from. Defaults to current directory.
    """
    three_digit_dirs = [
        d
        for d in os.listdir(base_directory)
        if os.path.isdir(os.path.join(base_directory, d)) and re.match(r"^\d{3}$", d)
    ]
    if len(three_digit_dirs) == 0:
        print(f"No three digit directories with three digits found in {base_directory}")
        three_digit_dirs = [base_directory]

    for three_digit_dir in three_digit_dirs:
        # print(f"*" * 80)
        # print(f"Processing directory: {three_digit_dir}")

        full_path = os.path.join(base_directory, three_digit_dir)

        checkpoint_dirs_last_epoch = []

        for epoch in range(100):

            # Find checkpoint directories in this three-digit directory
            checkpoint_dirs = [
                d
                for d in os.listdir(full_path)
                if os.path.isdir(os.path.join(full_path, d))
                and re.match(rf"checkpoint_{epoch}_n\d+", d)
            ]

            if not checkpoint_dirs:
                # print(
                #     f"Skipping directory {full_path} as no checkpoint directories were found."
                # )
                continue

            # Extract numbers from directory names and find the max
            max_checkpoint_number = max(
                int(re.search(rf"checkpoint_{epoch}_n(\d+)", d).group(1))
                for d in checkpoint_dirs
            )

            # Determine which directories to delete
            dirs_to_delete = [
                d
                for d in checkpoint_dirs
                if int(re.search(rf"checkpoint_{epoch}_n(\d+)", d).group(1))
                != max_checkpoint_number
            ]

            if delete_all:
                dirs_to_delete = checkpoint_dirs

            if not delete_all:
                assert len(dirs_to_delete) < len(
                    checkpoint_dirs
                ), "Error in deletion logic"

            if keep_only_last_epoch:
                if len(checkpoint_dirs) > 0:
                    dirs_to_delete += checkpoint_dirs_last_epoch
                    checkpoint_dirs_last_epoch = checkpoint_dirs

            print(f"Keeping:")
            for d in checkpoint_dirs:
                if d not in dirs_to_delete:
                    print(f"  {d}")

            print(f"Deleting:")
            for d in dirs_to_delete:
                print(f"  {d}")

            # Delete directories
            for dir_path in dirs_to_delete:
                try:
                    shutil.rmtree(os.path.join(full_path, dir_path))
                    print(f"Deleted: {dir_path}")
                except Exception as e:
                    print(f"Error deleting {dir_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean up checkpoint files, keeping only the highest numbered file."
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=".",
        help="Base directory to search for checkpoint files (default: current directory)",
    )

    parser.add_argument(
        "--delete_all",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--keep_only_last_epoch",
        default=False,
        action="store_true",
    )

    # Parse arguments
    args = parser.parse_args()
    cleanup_checkpoint_files(args.directory, args.delete_all, args.keep_only_last_epoch)
