# Improving Paratope and Epitope Prediction by Multi-Modal Contrastive Learning and Interaction Informativeness Estimation

A Python module for paratope and epitope prediction
![image](https://github.com/WangZhiwei9/MIPE/blob/main/Overview.jpg)

## Requirements

This project relies on specific Python packages to ensure its proper functioning. The required packages and their versions are listed in the `requirements.txt` file.

## Data

Dataset Files (pickle format) can be downloaded from: https://drive.google.com/drive/folders/1bvGZQnOs6XOA17NsiaZ4eVjvn94SOM3u?usp=drive_link

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
│      |─cvdata.pkl
│      |─testdata.pkl
├─requirements.txt
└─README.md
```

Training for the MIPE model with the dataset

```
python main.py
```

## License
DeepInterAware content and derivates are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Cite Us
Feel free to cite this work if you find it useful to you!

```
@inproceedings{ijcai2024p669,
  title     = {Improving Paratope and Epitope Prediction by Multi-Modal Contrastive Learning and Interaction Informativeness Estimation},
  author    = {Wang, Zhiwei and Wang, Yongkang and Zhang, Wen},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {6053--6061},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/669},
  url       = {https://doi.org/10.24963/ijcai.2024/669},
}
```

