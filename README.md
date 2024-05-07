# Improving Paratope and Epitope Prediction by Multi-Modal Contrastive Learning and Interaction Informativeness Estimation

A Python module for paratope and epitope prediction
![image](https://github.com/WangZhiwei9/MIPE/blob/main/Overview.jpg)

## Requirements

This project relies on specific Python packages to ensure its proper functioning. The required packages and their versions are listed in the `requirements.txt` file.

## Data

we compiled a dataset consisting of 626 binding antibody-antigen pairs, including their sequences, structures, and corresponding interaction maps.  All data is stored in the `alldata.pkl` file.

#### Note

Due to the large size of the `dataset.pkl` file, exceeding the 50MB size limit, it is not included in the current upload files. **Instead, only a subset of the data is provided as an example (`subdata.pkl`)**. Upon acceptance of the paper, we will upload the complete data file.

## Code

Our code files are packaged in zip format, and the directory structure is as follows.

```
MIPE/
├─code/
│  │  main.py
│  │  utils.py
│  |  model.py
│  |  NTXentLoss.py
│  |  CrossAttention.py
│  └─output_files/
│      |─modelsave
│      └─check_point
└─data/
│  └─dataset/
│      |─alldata.pkl
├─requirements.txt
└─README.md
```
