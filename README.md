https://www.kaggle.com/competitions/open-problems-multimodal/

## Goal of the Competition
The goal of this competition is to predict how DNA, RNA, and protein measurements co-vary in single cells as bone marrow stem cells develop into more mature blood cells. 

You will develop a model trained on a subset of 300,000-cell time course dataset of CD34+ hematopoietic stem and progenitor cells (HSPC) from four human donors at five time points generated for this competition by Cellarity, a cell-centric drug creation company.

In the test set, taken from an unseen later time point in the dataset, competitors will be provided with one modality and be tasked with predicting a paired modality measured in the same cell. The added challenge of this competition is that the test data will be from a later time point than any time point in the training data.

Your work will help accelerate innovation in methods of mapping genetic information across layers of cellular state. If we can predict one modality from another, we may expand our understanding of the rules governing these complex regulatory processes.

For the Multiome samples: given chromatin accessibility, predict gene expression.
For the CITEseq samples: given gene expression, predict protein levels.

## To study
- Cell Types: To help guide your analysis, we performed a preliminary cell type annotation based on the RNA gene expression using information from the following paper: https://www.nature.com/articles/ncb3493. Note, cell type annotation is an imprecise art, and the concept of assigning discrete labels to continuous data has inherent limitations. You do not need to use these labels in your predictions; they are primarily provided to guide exploratory analysis. In the data, there are the following cell types:
  - MasP = Mast Cell Progenitor
  - MkP = Megakaryocyte Progenitor
  - NeuP = Neutrophil Progenitor
  - MoP = Monocyte Progenitor
  - EryP = Erythrocyte Progenitor 
  - HSC = Hematoploetic Stem Cell 
  - BP = B-Cell Progenitor
- Competition announcement [twitter thread](https://twitter.com/dbburkhardt/status/1559304603933589504) 
   - Gene expression (rna)
- Pseudotime Algorithms
  - "In single-cell data science, dynamic processes have been modeled by so-called pseudotime algorithms that capture the progression of the biological process. Yet, generalizing these algorithms to account for both pseudotime and real time is still an open problem."
- Potentially useful tools
  - xarray/Dask
  - https://github.com/ewels/MultiQC
  - https://scverse.org/
  - Other [sc tools](https://www.kaggle.com/competitions/open-problems-multimodal/discussion/344816) 
- Research papers:
  - [recommended on Kaggle](https://www.kaggle.com/competitions/open-problems-multimodal/discussion/344686)
  - https://www.kaggle.com/competitions/open-problems-multimodal/discussion/344687
  - https://www.kaggle.com/competitions/open-problems-multimodal/discussion/344688
  - [Example notebooks](https://www.kaggle.com/competitions/open-problems-multimodal/discussion/344824) 
  - https://www.kaggle.com/competitions/open-problems-multimodal/discussion/346686


## Background

- Last year's (2021) [competition intro](https://twitter.com/MorseCell/status/1559583479158931459). See `2021 Competition Notes.md`.

## Notes


### CD34+ hematopoietic stem and progenitor cells (HSPCs)
- "Hematopoietic stem and progenitor cells (HSPCs) are a rare population of precursor cells that possess the capacity for self-renewal and multilineage differentiation."
- https://en.wikipedia.org/wiki/Hematopoietic_stem_cell


## Random ideas

Dna and Rna expression could be passed through [Unirep](https://github.com/ElArkk/jax-unirep) as a form of dimensionality reduction