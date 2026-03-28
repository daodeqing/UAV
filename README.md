# FIDSUS

The source code is for the paper: FIDSUS: Federated Intrusion Detection for Securing UAV Swarms

## Overview

The dynamic nature of UAV swarms, characterized by communication instability, heterogeneous nodes, and frequent topology changes, makes them vulnerable to network attacks. To address these challenges, we propose FIDSUS, a federated learning-based intrusion detection system. FIDSUS quantifies the similarity between UAVs' local feature extractors using an affinity matrix, facilitating knowledge sharing and enhancing the robustness of local models. By employing cross-round feature fusion, FIDSUS mitigates the forgetting problem caused by data heterogeneity, improving detection accuracy in complex scenarios. The global classifier, trained on fused feature representations, further boosts generalization performance. Experimental results demonstrate FIDSUS's superior performance compared to existing FL approaches, achieving an average accuracy improvement of 4% to 34% in large-scale scenarios. FIDSUS exhibits exceptional robustness and accuracy in dynamic client environments while maintaining competitive training efficiency.


## Dependencies

This project requires the following dependencies to be installed:

### Conda Channels

Make sure you have the following conda channels enabled:

- `pytorch`
- `nvidia`
- `defaults`

### Packages

The following packages are required:

- `pip==22`
- `pandas`
- `scikit-learn`
- `scipy`
- `ujson`
- `h5py`
- `seaborn`
- `matplotlib`

- `torch==2.0.1`
- `torchaudio`
- `torchtext`
- `torchvision`
- `calmsize`
- `memory-profiler`
- `opacus`
- `portalocker`
- `cvxpy`

### Installation

To install all dependencies, you can create a conda environment using the provided `env.yaml` file:

```bash
conda env create -f env.yaml
```

## Dataset

The dataset used for experimentation is the NSL-KDD and UNSW-NB15 dataset. We have pre-partitioned the USNW-NB15 dataset into 50 subsets according to a Dirichlet distribution to facilitate training. You can modify the partitioning method by adjusting the parameters in the `generate_unsw.py` file.

## Core Components

The core components of this project, which are central to the algorithms discussed in our paper, are implemented in the following files:


- **Client Implementation**: `system/flcore/clients/clientFIDSUS.py`
- **Server Implementation**: `system/flcore/servers/serverFIDSUS.py`

Additional algorithm (incremental, FIDSUS-compatible):

- **HGFIDSUS (Hierarchical-Game FIDSUS)**
	- Client: `system/flcore/clients/clientHGFIDSUS.py`
	- Server: `system/flcore/servers/serverHGFIDSUS.py`

- **SGEFIDSUS (Stackelberg-Game Evolutionary FIDSUS)**
	- Client: `system/flcore/clients/clientSGEFIDSUS.py`
	- Server: `system/flcore/servers/serverSGEFIDSUS.py`
	- Main idea: shared encoder + personalized head + Stackelberg pricing + bandwidth-aware auction

- **SkyGuardPFIDS**
	- Client: `system/flcore/clients/clientSkyGuardPFIDS.py`
	- Server: `system/flcore/servers/serverSkyGuardPFIDS.py`
	- Main idea: shared encoder + cluster adapter + personalized head + UAV-aware evolutionary participation + dropout-aware secure aggregation

## Running the Project

To run the project, navigate to the `system` directory and execute the following command:

```bash
bash run.sh
```

You can also run a specific algorithm/dataset from the `system` directory:

```bash
# UNSW-NB15
python main.py -algo HGFIDSUS -data UNSW -dev cuda

# NSL-KDD (num_classes will be auto-read from dataset/NSLKDD/config.json)
python main.py -algo HGFIDSUS -data NSLKDD -dev cuda

# SGEFIDSUS (same NPZ dataset format)
python main.py -algo SGEFIDSUS -data UNSW -dev cuda

# SkyGuardPFIDS
python main.py -algo SkyGuardPFIDS -data UNSW -dev cuda
```

If you want to add T-ITS CSV into this codebase's NPZ format, generate it first:

```bash
cd dataset
python generate_tits.py --csv /path/to/tits.csv --label-col label --output-dir TITS --num-clients 50 --niid True --partition dir
```

Then run:

```bash
cd system
python main.py -algo SGEFIDSUS -data TITS -dev cuda
python main.py -algo SkyGuardPFIDS -data TITS -dev cuda
```

