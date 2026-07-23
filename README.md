# HOT3D Hand Restoration Workspace

This repository contains the reproducible training and inference workspace for
the HOT3D single-frame hand-restoration experiment. The current training path
warps a HOT3D fisheye frame into the canonical C1 pinhole camera, renders the
annotated MANO hand in the same camera, builds a masked ControlNet condition,
and trains a ControlNet initialized from Stable Diffusion 1.5.

Large datasets, licensed MANO files, downloaded model weights, and experiment
outputs are intentionally excluded from Git.

## Repository layout

- `hand_restoration/`: dataset, condition construction, diffusion, inference,
  and visualization modules.
- `configs/hand_restoration/`: training, inference, and debug configurations.
- `train_hand_restorer.py`: ControlNet training entry point.
- `infer_hand_restorer.py`: command-line inference for one frame/checkpoint.
- `compare_hand_restoration_checkpoints.py`: interactive checkpoint comparison
  GUI.
- `smoke_test_hand_restoration.py`: preprocessing and optional diffusion test.
- `check_training_setup.py`: fast server/environment preflight check.
- `hot3d/`: pinned HOT3D Git submodule with two compatibility fixes.
- `README_hand_restoration.md`: implementation details and experiment notes.

The other top-level scripts support HOT3D undistortion, MANO/glove sequence
export, Blender import, glove rig construction, and compositing.

## Clone

Clone the repository together with the pinned HOT3D dependency:

```bash
git clone --recurse-submodules <YOUR_REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

## Create the environment

The environment is based on Miniforge/Conda and Python 3.10. PyTorch is
installed separately so that its CUDA wheel cannot be replaced by another
dependency.

```bash
conda env create -f environment.glove-hot3d.yml
conda activate glove-hot3d
python -m pip install -r requirements.glove-hot3d-gpu.txt
python -m pip install -r requirements.glove-hot3d.txt
```

`requirements.glove-hot3d-gpu.txt` reproduces the current CUDA 12.8 setup.
The server NVIDIA driver must support CUDA 12.8. If it does not, replace only
that file's PyTorch index and versions with an official compatible wheel set;
the project itself does not require a separately installed CUDA Toolkit.

The first diffusion run downloads `runwayml/stable-diffusion-v1-5` from
Hugging Face. Set `HF_HOME` to persistent server storage if the default home
directory is temporary:

```bash
export HF_HOME=/path/to/persistent/huggingface-cache
```

## Add external assets

The 3000-step configuration requires this exact relative layout:

```text
data/
  train_quest3/
    clip-000000.tar
mano_v1_2/
  models/
    MANO_LEFT.pkl
    MANO_RIGHT.pkl
```

Copy the original HOT3D clip tar without extracting it. Download MANO v1.2
from the official MANO site after accepting its license; MANO model files must
not be committed or redistributed. See `data/README.md` and
`mano_v1_2/README.md`.

## Verify and train

First verify paths, imports, submodule state, and CUDA:

```bash
python check_training_setup.py \
  --config configs/hand_restoration/tiny_overfit_shaded_3000.json \
  --require-cuda
```

Then verify one complete preprocessing sample:

```bash
python smoke_test_hand_restoration.py \
  --config configs/hand_restoration/tiny_overfit_shaded_3000.json
```

Run the current shaded-MANO tiny-overfit experiment for 3000 steps:

```bash
accelerate launch --num_processes 1 train_hand_restorer.py \
  --config configs/hand_restoration/tiny_overfit_shaded_3000.json
```

Checkpoints and `training_log.csv` are written to
`outputs/hand_restoration/tiny_overfit_shaded_frames10_20_3000/`. To resume
from a checkpoint, keep the same config and add:

```bash
--resume outputs/hand_restoration/tiny_overfit_shaded_frames10_20_3000/controlnet_step1500.pt
```

For checkpoint-by-checkpoint visual comparison:

```bash
python compare_hand_restoration_checkpoints.py
```

## Reproducibility notes

- Relative paths are resolved from the repository root. Launch commands there.
- The tiny-overfit config uses HOT3D clip 000000, right camera `1201-2`, right
  MANO, grayscale output, frames 10 through 20, and seed 7.
- Dataset files, MANO files, Hugging Face cache, generated media, and all
  checkpoints remain local and are covered by `.gitignore`.
- `requirements.glove-hot3d.txt` pins the direct Python dependencies used by
  the current working environment. CUDA/PyTorch is pinned separately.
