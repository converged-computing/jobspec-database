import argparse
import os

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--save_dir", type=str, required=True)
    args.add_argument("--output_dir", type=str, required=True)
    args.add_argument("--delete", action="store_true")

    config = args.parse_args()

    print(config)

    # Check if output_dir exists, if not create it
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    for filename in os.listdir(config.save_dir):
        if filename.endswith(".npz"):
            new_filename = filename.split("_")[-1]
            new_filename = str(int(new_filename.split(".")[0])) + ".npz"
            os.rename(
                os.path.join(config.save_dir, filename),
                os.path.join(config.output_dir, new_filename),
            )
            print("Renamed {} to {}".format(filename, new_filename))
    
    # Delete the save_dir
    if config.delete:
        os.rmdir(config.save_dir)
