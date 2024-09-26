#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --time=04:30:00
##SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
##SBATCH --mem=8G
##SBATCH --output "slurm/out/shinobi_frames_%A.out"

module load python/3.7.7

python make_dataset.py --shinobi --filename shinobi_frames_128 --resolution 128 128

[sana4471@beluga4 shinobi_data_set]$ vim make_dataset.py 
[sana4471@beluga4 shinobi_data_set]$ cat make_dataset.py
import os
import glob
import argparse
import urllib.request
import h5py
from tqdm import tqdm
from presses_to_frames import presses_to_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--moving_mnist",
        action="store_true",
        help="Download the data for the moving MNIST dataset.",
    )
    parser.add_argument(
        "--shinobi",
        action="store_true",
        help="Generate the frames from the key presses logs of the CNeuromod "
        "shinobi dataset. It is assumed that prior to doing that you "
        "have already installed the shinobi dataset in the data folder.",
    )
    parser.add_argument(
        "--resolution",
        nargs="+",
        type=int,
        default=(64, 64),
        help="Resolution to resample the frames to.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="shinobi_frames",
        help="Name foe the hdf5 file to store the results in.",
    )
    args = parser.parse_args()

    if args.moving_mnist:
        os.makedirs("data/mnist", exist_ok=True)
        print("Downloading the MNIST dataset.")
        urllib.request.urlretrieve(
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "data/mnist/train-images-idx3-ubyte_2.gz",
        )
        print("MNIST dataset downloaded, downloding the moving_MNIST test set.")
        os.makedirs("data/moving_mnist", exist_ok=True)
        urllib.request.urlretrieve(
            "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy",
            "data/moving_mnist/mnist_test_seq_2.npy",
        )
        print("All done.")

    if args.shinobi:
        import retro

        if not os.path.isdir("/project/rrg-pbellec/sana4471/movie_decoding_sa/data/cneuromod.shinobi.fmriprep"):
            raise FileNotFoundError(
                "data/shinobi directory not found. Install the CNeuromod "
                "shinobi dataset before running this."
            )

        stim_file_template = "/project/rrg-pbellec/sana4471/movie_decoding_sa/data/cneuromod.shinobi.fmriprep/sourcedata/shinobi/sub-*/ses-*/gamelogs/*.bk2"
        retro.data.Integrations.add_custom_path(
            "/project/rrg-pbellec/sana4471/movie_decoding_sa/data/cneuromod.shinobi.stimuli"
        )
        emulator = retro.make(
            "ShinobiIIIReturnOfTheNinjaMaster-Genesis",
            state=retro.State.NONE,
            use_restricted_actions=retro.Actions.ALL,
            players=1,
            inttype=retro.data.Integrations.CUSTOM_ONLY
        )
        

        out_path = os.path.join("data", os.path.splitext(args.filename)[0] + ".hdf5")
        with h5py.File(out_path, "a") as h5file:
            for path in tqdm(glob.glob(stim_file_template)):
                sub = "sub-" + path.split("sub-")[1][:2]
                print(path)
                print(path.split("Genesis_Level"))
                #level = "Level" + path.split("Genesis_Level")[1][:3]
                level ="Level" + path.split("level-")[1][0]
                print(level)
                emulator.reset()
                images = presses_to_frames(path, emulator, size=args.resolution)
                print(os.path)
                filename = os.path.splitext(os.path.split(path)[1])[0]
                grp = h5file.require_group(f"/{sub}/{level}")
                grp.create_dataset(filename, data=images)


if __name__ == "__main__":

    #print('sana')
    main()

