# Inferring Gene Regulatory Networks via Directed Graph Contrastive Representation Learning

This repository contains the code for the paper **"Inferring Gene Regulatory Networks via Directed Graph Contrastive Representation Learning"**. 

## Datasets

The experiments in the paper use two datasets:

1. **DREAM5 Dataset**  
   The DREAM5 dataset is a well-known benchmark used in gene regulatory network inference. You can download the dataset from the following link:  
   [DREAM5 Dataset](https://www.synapse.org/Synapse:syn2787209/wiki/70349)

2. **Single-cell Gene Expression Dataset**  
   This dataset is derived from single-cell RNA-seq experiments. You can download the dataset from the **GENELink** repository:
   [Single-cell Gene Expression Dataset](https://github.com/zpliulab/GENELink)


To change the dataset, you need to delete the “processed” folder in the corresponding folder. Use “train.py” and “load_data.py” if inferring DREAM5, or “train_SC.py” and “load_data_SC.py” if inferring single-cell.
