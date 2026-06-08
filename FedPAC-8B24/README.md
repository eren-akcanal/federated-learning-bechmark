
# FedPAC: Taming Preconditioner Drift for Federated Second-Order Optimization

This repository provides an implementation of **FedPAC (Federated Preconditioner Alignment and Correction)** — a framework designed to **stabilize and accelerate federated second-order (or quasi-second-order) optimization** under **Non-IID** data, and to make modern optimizers such as **Sophia / Muon / SOAP** work reliably in federated learning.

**Paper:** *Taming Preconditioner Drift: Unlocking the Potential of Second-Order Optimizers for Federated Learning on Non-IID Data* (ICML under review)

---

## Motivation

Second-order methods rely on curvature-related states (e.g., preconditioners). In federated learning, each client updates these states on its own local (often Non-IID) data, causing **preconditioner drift** across clients. When the server performs model averaging, the updates are mixed under **inconsistent geometries**, which can slow convergence or even destabilize training.

**FedPAC** addresses this by:
- **Alignment:** aggregate client preconditioners on the server to form a global reference and broadcast it to clients (warm-start).
- **Correction:** locally mix the update directions guided by the local and global preconditioners to suppress long-term drift accumulation.

---

## Features

- Supports **FedPAC variants** built on top of:
  - **Sophia**  → `FedSophia`
  - **Muon**    → `FedMuon`
  - **SOAP**    → `FedSoap`
- Includes several baselines (e.g., `FedAvg`, `SCAFFOLD`, `FedCM`, and local-only variants).
- Ray-based simulation for scalable multi-client federated training.

---

## Repository Structure

- `main_FedPAC.py`  
  Main entry point for federated training (Ray parallel simulation).
- `README.md`  
  Project documentation.

---

## Installation

Create an environment and install dependencies (adjust to your setup):

```bash
pip install -r requirements.txt
````

Typical requirements include: Python, PyTorch/torchvision, numpy, matplotlib, tensorboardX, filelock, ray, etc.

> Notes:
>
> * The code uses implementations of **SOAP** and **SophiaG** internally.
> * Optional: `peft` is used if you enable LoRA for ViT/Swin backbones.

---

## Datasets & Non-IID Partitioning

### Supported datasets

The script selects datasets via `--data_name` (e.g., `CIFAR10`, `CIFAR100`, and `imagenet` where the loader corresponds to **Tiny-ImageNet** in this codebase).

### Non-IID splits

Non-IID partitions are generated using a **Dirichlet** distribution controlled by:

* `--alpha_value`: smaller values mean stronger heterogeneity (more Non-IID).

The split indices are cached to improve reproducibility.

---

## Quick Start

Below are recommended commands matching common settings used in the paper (e.g., 100 clients, 10% participation, batch size 50, local steps `K=50`, 300 rounds).
FedPAC’s correction strength is controlled by `--alpha` in code (paper notation: **β**). A typical default is **0.5**.

### 1) FedPAC SOAP (`--alg FedSoap`)

```bash
python main_FedPAC.py \
  --alg FedSoap \
  --lr 3e-3 \
  --data_name CIFAR100 \
  --alpha_value 0.05 \
  --alpha 0.5 \
  --epoch 301 \
  --CNN resnet18 \
  --E 5 \
  --batch_size 50 \
  --num_workers 100 \
  --selection 0.1 \
  --K 50 \
  --pix 32 \
  --gpu 0 \
  --num_gpus_per 0.1 \
  --normalization BN \
  --preprint 10
```

### 2) FedPAC Muon (`--alg FedMuon`)

```bash
python main_FedPAC.py \
  --alg FedMuon \
  --lr 3e-2 \
  --data_name CIFAR100 \
  --alpha_value 0.05 \
  --alpha 0.5 \
  --epoch 301 \
  --CNN resnet18 \
  --E 5 \
  --batch_size 50 \
  --num_workers 100 \
  --selection 0.1 \
  --K 50 \
  --pix 32 \
  --gpu 0 \
  --num_gpus_per 0.1 \
  --normalization BN \
  --preprint 10
```

### 3) FedPAC Sophia (`--alg FedSophia`)

```bash
python main_FedPAC.py \
  --alg FedSophia \
  --lr 3e-4 \
  --rho 0.1 \
  --data_name CIFAR100 \
  --alpha_value 0.05 \
  --alpha 0.5 \
  --epoch 301 \
  --CNN resnet18 \
  --E 5 \
  --batch_size 50 \
  --num_workers 100 \
  --selection 0.1 \
  --K 50 \
  --pix 32 \
  --gpu 0 \
  --num_gpus_per 0.1 \
  --normalization BN \
  --preprint 10
```

---

## Key Arguments

* `--alg`
  Algorithm name (e.g., `FedSoap`, `FedMuon`, `FedSophia`, plus baselines).
* `--lr`
  Learning rate.
* `--epoch`
  Number of communication rounds (or total global rounds).
* `--E`
  Local epochs per round.
* `--K`
  Local step budget (used to limit local updates).
* `--alpha_value`
  Dirichlet parameter controlling Non-IID heterogeneity.
* `--selection`
  Fraction of clients participating per round.
* `--alpha`
  FedPAC correction strength (paper: **β**), commonly set to `0.5`.
* `--CNN`
  Backbone model (e.g., `resnet18`, ViT/Swin variants if provided).
* `--pix`
  Input resolution (e.g., `32` for CIFAR, `224` for ViT/Swin).
* `--lora`, `--r`
  Enable LoRA and set rank (optional).
* `--num_workers`
  Number of total clients.
* `--gpu`, `--num_gpus_per`
  GPU selection and Ray GPU allocation per worker.

---

## Logging & Outputs

The training script typically:

* writes logs to `./log/*.txt`
* records metrics with TensorBoard (`tensorboardX`)
* saves curves/arrays to `./plot/*.npy` (e.g., accuracy, loss)

---

## Reproducibility Tips

* Fix seeds (if exposed in your config) and keep the cached partition indices.
* Report `alpha_value`, `selection`, `E`, and `K` since they strongly affect FL behavior.
* For fair comparison, match the same backbone and preprocessing (`--pix`, normalization).

---

## Citation

If you use this codebase, please cite:

```bibtex
@article{fedpac2026,
  title   = {Taming Preconditioner Drift: Unlocking the Potential of Second-Order Optimizers for Federated Learning on Non-IID Data},
  author  = {Anonymous Authors},
  journal = {Under review},
  year    = {2026}
}
```

---

## License

Add your license here (e.g., MIT / Apache-2.0) or see `LICENSE` if provided.

---

```
```
