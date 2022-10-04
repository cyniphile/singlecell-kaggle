https://www.kaggle.com/competitions/open-problems-multimodal/

## Goal of the Competition
The goal of this competition is to predict how DNA, RNA, and protein measurements co-vary in single cells as bone marrow stem cells develop into more mature blood cells. 

You will develop a model trained on a subset of 300,000-cell time course dataset of CD34+ hematopoietic stem and progenitor cells (HSPC) from four human donors at five time points generated for this competition by Cellarity, a cell-centric drug creation company.



For the Multiome samples: given chromatin accessibility, predict gene expression.
For the CITEseq samples: given gene expression, predict protein levels.

## To study
~~- Competition announcement [twitter thread](https://twitter.com/dbburkhardt/status/1559304603933589504)~~
   - Paper spun off from last year: https://twitter.com/satijalab/status/1498319810459062287
   - [video about last year](https://www.youtube.com/watch?v=Fm-MDpPo85c&ab_channel=OpenProblemsinSingle-CellAnalysis)
   - [winning team explanation videos from last year](https://drive.google.com/file/d/1aQss-KyfYlzdrBQcH5joiXMlTwpG5gdf/view)
   - [site and explanations of assays from last year](https://openproblems.bio/neurips_docs/data/about_multimodal/)
- Cell Types: To help guide your analysis, we performed a preliminary cell type annotation based on the RNA gene expression using information from the following paper: https://www.nature.com/articles/ncb3493. Note, cell type annotation is an imprecise art, and the concept of assigning discrete labels to continuous data has inherent limitations. You do not need to use these labels in your predictions; they are primarily provided to guide exploratory analysis. In the data, there are the following cell types:
  - MasP = Mast Cell Progenitor
  - MkP = Megakaryocyte Progenitor
  - NeuP = Neutrophil Progenitor
  - MoP = Monocyte Progenitor
  - EryP = Erythrocyte Progenitor 
  - HSC = Hematoploetic Stem Cell 
  - BP = B-Cell Progenitor
- Probably good to level up your detail in the understanding of the dogma of molecular bio, including epigenetics, Post-transcriptional modification, Transcription factors, Gene expression, and the correlation of rna and protein.
- [The triumphs and limitations of computational methods for scRNA-seq](https://www.nature.com/articles/s41592-021-01171-x)
- [Book on this subject](https://github.com/theislab/cross-modal-single-cell-best-practices/) 
- Pseudotime Algorithms
  - "In single-cell data science, dynamic processes have been modeled by so-called pseudotime algorithms that capture the progression of the biological process. Yet, generalizing these algorithms to account for both pseudotime and real time is still an open problem."
- Potentially useful tools
  - xarray/Dask
  - https://github.com/ewels/MultiQC
  - https://scverse.org/, https://www.youtube.com/channel/UCpsvsIAW3R5OdftJKKuLNMA
  - Other [sc tools](https://www.kaggle.com/competitions/open-problems-multimodal/discussion/344816) 
  - https://kipoi.org/
- Research papers:
  - [recommended on Kaggle](https://www.kaggle.com/competitions/open-problems-multimodal/discussion/344686)
  - https://www.kaggle.com/competitions/open-problems-multimodal/discussion/344687
  - https://www.kaggle.com/competitions/open-problems-multimodal/discussion/344688
  - [Example notebooks](https://www.kaggle.com/competitions/open-problems-multimodal/discussion/344824) 
  - https://www.kaggle.com/competitions/open-problems-multimodal/discussion/346686


## Background

- Last year's (2021) [competition intro](https://twitter.com/MorseCell/status/1559583479158931459). See `2021 Competition Notes.md`.

## Notes

### Multivariate regression

For a simple linear model, it seems to be equivalent to running $n$ regressions when you want to predict and length $n$ vector, but would like to worth through the math?

A few sklearn methods are naturally multi-output (`LinearRegression` (and related) `KNeighborsRegressor DecisionTreeRegressor RandomForestRegressor`), and for the rest there is a wrapper `MultiOutputRegressor(model)` that runs $n$ single-output regressions using any single-output model. 



### CD34+ hematopoietic stem and progenitor cells (HSPCs)
- "Hematopoietic stem and progenitor cells (HSPCs) are a rare population of precursor cells that possess the capacity for self-renewal and multilineage differentiation."
- https://en.wikipedia.org/wiki/Hematopoietic_stem_cell


## Random ideas

- Dna and Rna expression could be passed through [Unirep](https://github.com/ElArkk/jax-unirep) as a form of dimensionality reduction
- Someone notes that most of the donors are male, but one is female. Brainstorm other characteristics and features to extract
- Since we are working with count data (albeit normalized), I'm thinking poisson regression might lend a more powerful prior. 
- Make sure to account for promoter regions/transcription factors as genes  

