This is an official PyTorch implementation of EMNLP 2025 paper "An Efficient Quantum Graph Fusion Network for Multimodal Rumor Detection."

This is a raw version at the moment; a tweaked version will be released online after the paper is accepted.

# Dataset
  The datasets used in the experiments were based on the two publicly available Weibo and PHEME datasets released by Song et al. (2019) and Zubiaga et al. (2017), and the preprocess process was based on the work by Yuan et al. (2019):
  
  Changhe Song, Cheng Yang, Huimin Chen, Cunchao Tu, Zhiyuan Liu, and Maosong Sun. Ced: Credible early detection of social media rumors. IEEE Transactions on Knowledge and Data Engineering, 33(8):3035–3047, 2019.
  
  Arkaitz Zubiaga, Maria Liakata, and Rob Procter. Exploiting context for rumour detection in social media. InInternational Conference on Social Informatics, pages 109–123. Springer, 2017.
  
  Chunyuan Yuan, Qianwen Ma, Wei Zhou, Jizhong Han, and Songlin Hu. Jointly embedding the local and global relations of heterogeneous graph for rumor detection. InICDM, pages 796–805. IEEE, 2019.

# Dependencies
  
- Python 3.8.0
- Pytorch 1.7.1 

- (1) Create Conda Environment

```bash
conda create --name  EQGFNet: python=3.8
conda activate  EQGFNet:
```

- (2) Install Dependencies

```bash
cd  EQGFNet
pip install -r requirements.txt
```  


- (3) Train and test
 
 ```
 cd ./graph_part
 python pheme_threemodal.py 
 ```
