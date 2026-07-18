# CLIMB: Consistent Longitudinal Individual Multi-Modal Cortical Parcellation for Infant Population

## Overview

Individual multi-modal cortical parcellation in infants is crucial for 
comprehensive and precise early comparative, longitudinal, and downstream 
analyses from both structural and functional perspectives. However, 
existing methods are predominantly designed for cross-sectional adult 
studies, thus failing to handle topological shifts caused by high 
inter-infant variability, elevated data noise, and longitudinal study demands.
Moreover, their multi-modal integration typically relies on global weights, 
handcrafted rules, or naive concatenation, failing to fully exploit 
cross-modal complementarity.

To address these issues, we propose CLIMB, a Consistent Longitudinal 
Individual Multi-modal Brain parcellation method for infants. 
Specifically, building upon conventional losses, CLIMB also guides 
optimization via three additional tailored losses: a feature 
consistency loss matching the feature profiles of individual 
parcels with their same-label group-level counterparts to 
correct topological misalignments; a longitudinal consistency 
loss constraining intra-subject longitudinal similarities 
across time points to capture continuous developmental 
changes; and a dynamic spatially weighted homogeneity 
loss progressively optimizing homogeneity while avoiding 
noise-induced fragmentation. For multi-modal integration, 
CLIMB adaptively captures complementary information via 
multi-layer graph mutual learning and vertex-level modality weighting.

Experiments on the Baby Connectome Project dataset demonstrate 
that CLIMB surpasses state-of-the-art methods in overall 
metric performance and stably surpasses the group-level 
alternative on the two downstream tasks, while yielding more 
consistent and realistic longitudinal developmental 
trajectories compared to cross-sectional methods.

## Environment Setup

The experiments were conducted with the following environment:

-   Python 3.10
-   PyTorch 2.10.0
-   CUDA 12.8
-   GPU: NVIDIA RTX 5070 Ti

Install the required packages:

``` bash
pip install -r requirements.txt
```

## Dataset Preparation

CLIMB is evaluated on the longitudinal Baby Connectome Project (BCP)
dataset.

The input data consist of multimodal cortical features, including
functional connectivity networks (FCN) and morphological similarity
networks (MSN), together with cortical surface topology information and
reference parcellation maps.

The required inputs include:

-   Group-level multimodal features and reference parcellation maps.
-   Individual longitudinal FCN and MSN features across multiple
    developmental time points.
-   Cortical vertex coordinates and 1-ring neighborhood adjacency
    information.

The data paths should be specified through the arguments in `Main.py`.

## Training

Before training, modify the dataset paths in `Main.py` according to your
data organization.

Run:

``` bash
python Main.py
```

The training procedure jointly optimizes the proposed objectives,
including reference prior similarity, dynamic spatial homogeneity,
spatial continuity, feature consistency, and longitudinal consistency
losses.

Default training settings:

-   Optimizer: AdamW
-   Training steps: 1000
-   Device: CUDA GPU

## Testing

After training, the saved model parameters can be loaded to generate
longitudinal individual cortical parcellation maps.

The testing procedure is automatically executed in `Main.py` after
loading the trained model.

## Main Components

The repository structure is organized as follows:

    CLIMB/
    ├── code/
    │   ├── Main.py
    │   ├── Dataloader.py
    │   ├── Model.py
    │   ├── Loss.py
    │   └── Trainer.py
    │
    ├── result/
    │   ├── fine-grained_functional_map/
    │   ├── in-house_multimodal_map/
    │   └── MMP/
    │
    ├── README.md
    └── requirements.txt

### Code

The `code/` folder contains the implementation of CLIMB:

-   `Dataloader.py`: Loads multimodal cortical features and constructs
    graph-based inputs.

-   `Model.py`: Implements the CLIMB network, including multi-layer
    graph-based multimodal learning and vertex-level modality weighting.

-   `Loss.py`: Implements optimization objectives, including reference prior similarity, dynamic spatial homogeneity,
spatial continuity, feature consistency, and longitudinal consistency
losses.

-   `Trainer.py`: Provides the training and inference pipeline.

-   `Main.py`: Entry point for training and testing.

### Results

The `result/` folder contains individual cortical parcellation results
obtained using different group-level reference atlases.

Specifically:

-   `fine-grained_functional_map/`: Contains individual parcellation
    results generated using the fine-grained functional map as the
    reference prior.

-   `in-house_multimodal_map/`: Contains individual parcellation results
    generated using the in-house multimodal map as the reference prior.

-   `MMP/`: Contains individual parcellation results generated using the
    MMP map as the reference prior.

Each folder includes the corresponding individual parcellation results
produced by CLIMB under the respective group-level reference prior.


## Reproducibility

The provided implementation supports reproducing the longitudinal
individual cortical parcellation experiments described in the paper.

After training, the model generates individual longitudinal cortical
parcellation maps through the testing pipeline.
