INSE 6220 Course Project – Reproduction of Truncated Affinity Maximization (TAM) on the Amazon Dataset

Student: Ismail Mzouri
Course: INSE 6220 – Advanced Statistical Approaches to Quality
Institution: Concordia University
Semester: Fall 2025

This repository contains my course project reproduction of the paper
“Truncated Affinity Maximization: One-class Homophily Modeling for Graph Anomaly Detection” (NeurIPS 2023) by Hezhe Qiao and Guansong Pang.

The objective of this project is to reproduce, analyze, and validate the TAM methodology on a real-world graph anomaly detection dataset, following the experimental protocol of the original paper.

The reproduction focuses exclusively on the Amazon co-review network dataset, which satisfies the INSE 6220 course requirements. Results on other datasets are reported only for contextual comparison and are taken directly from the original paper.

This repository includes a complete, reproducible pipeline, statistical analysis, PCA exploration, baseline comparison, and all figures used in the final IEEE-formatted report.

My contributions:

Reproduced the TAM pipeline on the Amazon dataset

Achieved AUROC = 0.706 and AUPRC = 0.263, consistent with the original paper

Implemented an Isolation Forest baseline for comparison

Conducted Principal Component Analysis (PCA) on the Amazon dataset

Generated all plots and figures used in the report

Provided a fully reproducible notebook-based workflow

To reproduce the results, first install the dependencies using the provided requirements file.

Install dependencies:
pip install -r requirements_current.txt

You can then run the reproduction using one of the following options.

Notebook-based execution (recommended):
jupyter notebook Amazon_TAM.ipynb

Command-line execution:
python train.py --dataset Amazon

The expected output should be approximately:
AUROC ≈ 0.706
AUPRC ≈ 0.263

Training logs are saved in:
tam_amazon_log.txt

Reproduction results on the Amazon dataset:

Method: Isolation Forest
AUROC: 0.562
AUPRC: 0.137
Description: Attribute-only baseline

Method: TAM (This Reproduction)
AUROC: 0.706
AUPRC: 0.263
Description: Structure-aware graph anomaly detection

Key observations:

TAM improves AUROC by 14.4 percentage points over the baseline

PCA shows that approximately 20 principal components capture 90% of variance

Normal nodes form dense clusters while anomalies appear scattered

Results validate the one-class homophily hypothesis on real-world fraud data

Repository structure:

Amazon_TAM.ipynb: Complete reproduction notebook

tam_amazon_log.txt: Training logs

figures/: PCA plots and performance visualizations

model.py: LAMNet implementation

train.py: TAM training pipeline

utils.py: Graph preprocessing utilities

degree_nsgt.py: NSGT implementation

raw_affinity.py: Local affinity computation

dis_statistic.py: Distance statistics analysis

data/: Amazon dataset files

Dataset information:

Dataset: Amazon (UPU)
Nodes: 10,244
Edges: 175,608
Attributes: 25
Anomalies: 693 (6.66%)

Nodes represent products, edges represent co-reviewed products, and anomalies correspond to products associated with fraudulent reviews.

Methodology summary:

Local node affinity measures similarity between nodes and their neighbors

LAMNet (2-layer GCN) learns representations by maximizing local affinity

NSGT probabilistically removes non-homophily edges

An ensemble is built using multiple truncation depths and runs

Hyperparameters used for the Amazon dataset:

GCN layers: 2

Hidden dimension: 64

Learning rate: 1e-5

Epochs: 500

Truncation depth K: 4

Ensemble size T: 3

Regularization λ: 0 (real anomalies)

All code, logs, figures, and configurations required to reproduce the Amazon dataset results are included in this repository. The project was executed in a course-level computing environment and follows the original paper’s experimental protocol for the Amazon dataset.

This work builds upon the official implementation of:
Truncated Affinity Maximization: One-class Homophily Modeling for Graph Anomaly Detection
NeurIPS 2023
https://arxiv.org/pdf/2306.00006.pdf

The original paper reports results on multiple benchmark datasets (BlogCatalog, ACM, Facebook, Reddit, YelpChi, Amazon-all, YelpChi-all, T-Finance, OGB-Protein). Only the Amazon dataset is reproduced here.

If you use this work, please cite the original paper:

@inproceedings{qiao2023truncated,
title={Truncated Affinity Maximization: One-class Homophily Modeling for Graph Anomaly Detection},
author={Qiao, Hezhe and Pang, Guansong},
booktitle={Advances in Neural Information Processing Systems},
year={2023}
}

This repository is submitted as part of INSE 6220 – Advanced Statistical Approaches to Quality at Concordia University and is intended for educational and reproducibility purposes only.
