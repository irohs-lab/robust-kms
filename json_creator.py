import os
import json
import argparse


def create_json_from_checkpoints(
    root_dir,
    output_file="models.json"
):
    results = []

    for dirpath, dirnames, filenames in os.walk(root_dir):

        for file in filenames:

            print(
                f"Checking file: {file} "
                f"in {dirpath}"
            )

            if file.endswith(".pt"):

                full_path = os.path.join(
                    dirpath,
                    file
                )

                # experiment name = folder name
                experiment = os.path.basename(
                    dirpath
                )

                results.append({
                    "experiment": experiment,
                    "path": full_path
                })

    output = {
        "root_dir": root_dir,
        "num_experiments": len(results),
        "results": results
    }

    with open(output_file, "w") as f:

        json.dump(
            output,
            f,
            indent=2
        )

    print(
        f"Saved {len(results)} "
        f"entries to {output_file}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=None
    )

    args = parser.parse_args()

    # auto naming if not provided
    if args.output_file is None:

        args.output_file = (
            f"{os.path.basename(args.root_dir)}.json"
        )

    create_json_from_checkpoints(
        args.root_dir,
        args.output_file
    )
