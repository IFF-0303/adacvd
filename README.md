# AdaCVD

ToDo: Introduction.

## 🛠️ Installation

This project uses `micromamba` — a fast, minimal installer for `conda` environments. It's a lightweight alternative to `conda` that supports the same environment and package management commands. If you're already using `conda`, you can adapt the steps accordingly, but we recommend `micromamba` for faster setups. If you don’t have `micromamba` installed, follow the [official installation guide](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

### 1. Create and Activate the Environment

```bash
micromamba create -n adacvd python=3.10
micromamba activate adacvd
```

### 2. Install PyTorch with CUDA Support

```bash
micromamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
> Adjust `pytorch-cuda=12.1` if your system uses a different CUDA version.

### 3. Install Remaining Dependencies

```bash
micromamba install -f environment.yaml
```

### 4. Verify CUDA Availability

```bash
python -c "import torch; assert torch.cuda.is_available()"
```

### 5. Install the Project in Editable Mode

```bash
pip install -e .
```

## Data

ToDo.

## Model Training & Inference

### Training

ToDo.

### Inference

ToDo.

## Baselines

ToDo.

## Evaluation

ToDo.

##

