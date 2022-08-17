https://www.kaggle.com/competitions/open-problems-multimodal/

## Goal of the Competition
The goal of this competition is to predict how DNA, RNA, and protein measurements co-vary in single cells as bone marrow stem cells develop into more mature blood cells. 

You will develop a model trained on a subset of 300,000-cell time course dataset of CD34+ hematopoietic stem and progenitor cells (HSPC) from four human donors at five time points generated for this competition by Cellarity, a cell-centric drug creation company.

In the test set, taken from an unseen later time point in the dataset, competitors will be provided with one modality and be tasked with predicting a paired modality measured in the same cell. The added challenge of this competition is that the test data will be from a later time point than any time point in the training data.

Your work will help accelerate innovation in methods of mapping genetic information across layers of cellular state. If we can predict one modality from another, we may expand our understanding of the rules governing these complex regulatory processes.

## To study

- Competition announcement [twitter thread](https://twitter.com/dbburkhardt/status/1559304603933589504) 
- ATAC data - Assay of Transcription available chromatin
- GEX data - gene expression
  - https://support.10xgenomics.com/single-cell-multiome-atac-gex/software/visualization/latest/tutorial-navigation
- "In single-cell data science, dynamic processes have been modeled by so-called pseudotime algorithms that capture the progression of the biological process. Yet, generalizing these algorithms to account for both pseudotime and real time is still an open problem."
- CD34+ hematopoietic stem and progenitor cells (HSPCs)
- Potentially useful tools
  - https://github.com/ewels/MultiQC
  - https://scverse.org/


## Background

- Last year's (2021) [competition intro](https://twitter.com/MorseCell/status/1559583479158931459). See `2021 Competition Notes.md`.