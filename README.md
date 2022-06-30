# SEA - Graph Shell Attention

Implementation of Graph Shell Attention.

# Overview

* ```/configs``` contains the various parameters used in the ablation and SOTA comparison studies.
* ```DataLoader.py```loads and stores information about the graph being used
* ```SEAGNN.py``` Implementation of the SEA-GNN architecture
* ```SEAAGG.py``` Implementation of the SEA-Aggregated architecture
* ```TBase.py``` Implementation of the vanilla transformer-based GNN w/ Graph Transformer Layers (Dwivedi et al.)
* ```main.py``` main file; containing train+eval procedures

# Datasets
### Download datasets
* SBM Pattern: 
  [SBM_PATTERN:zip](https://www.dropbox.com/s/qvu0r11tjyt6jyb/SBM_PATTERN.zip)
  <br/>extract files to ```/dataset/SBMs/PATTERN/*```
* ZINC:
  [molecules_zinc_full.zip](https://www.dropbox.com/s/grhitgnuuixoxwl/molecules_zinc_full.zip)
  <br/>extract files to ```/dataset/molecules/zinc_full/*```
* ogbg-molhiv
  download via ogb 
  ```python 
  from ogb.graphproppred import DglGraphPropPredDataset
  dataset = DglGraphPropPredDataset(name = 'ogbg-molhiv')
  ``` 
  will automatically create ```/dataset/ogbg-molhiv/*```

# Running script
run main.py with a config file from ```/configs```. Model configuration + dataset are read from config files.<br/>
```
python main.py --config zinc_seagnn
```