https://www.kaggle.com/competitions/open-problems-multimodal/
## How To Run
1. 
```sh
pyenv install 3.10.6
```

```sh
brew install poetry
# Make a local `.venv` directory
poetry config virtualenvs.create false --local
poetry install
```

If poetry is giving you issues there is also a `requirements.txt` file available for standard installation of packages into a virtualenv.

2. Download [the dataset](https://www.kaggle.com/competitions/open-problems-multimodal/data) and extract to `data/original`.
3. Download [sparse dataset](https://www.kaggle.com/datasets/fabiencrom/multimodal-single-cell-as-sparse-matrix) and extract to `data/sparse`

4. Run basic tests `./test.sh`

5. Check out run data: `prefect orion start`


## To study
~~- Competition announcement [twitter thread](https://twitter.com/dbburkhardt/status/1559304603933589504)~~
   - Paper spun off from last year: https://twitter.com/satijalab/status/1498319810459062287
- Cell Types: To help guide your analysis, we performed a preliminary cell type annotation based on the RNA gene expression using information from the following paper: https://www.nature.com/articles/ncb3493. Note, cell type annotation is an imprecise art, and the concept of assigning discrete labels to continuous data has inherent limitations. You do not need to use these labels in your predictions; they are primarily provided to guide exploratory analysis. In the data, there are the following cell types:
  - MasP = Mast Cell Progenitor
  - MkP = Megakaryocyte Progenitor
  - NeuP = Neutrophil Progenitor
  - MoP = Monocyte Progenitor
  - EryP = Erythrocyte Progenitor 
  - HSC = Hematoploetic Stem Cell 
  - BP = B-Cell Progenitor
- Probably good to level up your detail in the understanding of the dogma of molecular bio, including epigenetics, Post-transcriptional modification, Transcription factors, Gene expression, and the correlation of rna and protein.
- Pseudotime Algorithms
  - "In single-cell data science, dynamic processes have been modeled by so-called pseudotime algorithms that capture the progression of the biological process. Yet, generalizing these algorithms to account for both pseudotime and real time is still an open problem."
- Potentially useful tools
  - xarray/Dask
  - https://github.com/ewels/MultiQC
  - https://scverse.org/, https://www.youtube.com/channel/UCpsvsIAW3R5OdftJKKuLNMA
  - Other [sc tools](https://www.kaggle.com/competitions/open-problems-multimodal/discussion/344816) 
  - https://kipoi.org/

## Notes

### Multivariate regression

For a simple linear model, it seems to be equivalent to running $n$ regressions when you want to predict and length $n$ vector, but would like to worth through the math?

A few sklearn methods are naturally multi-output (`LinearRegression` (and related) `KNeighborsRegressor DecisionTreeRegressor RandomForestRegressor`), and for the rest there is a wrapper `MultiOutputRegressor(model)` that runs $n$ single-output regressions using any single-output model. 

### CD34+ hematopoietic stem and progenitor cells (HSPCs)
- "Hematopoietic stem and progenitor cells (HSPCs) are a rare population of precursor cells that possess the capacity for self-renewal and multilineage differentiation."
- https://en.wikipedia.org/wiki/Hematopoietic_stem_cell


## Engineering Notes
 
- On visualization:
  - "It's time to stop making t-SNE & UMAP plots. In a new preprint w/ Tara Chari we show that while they display some correlation with the underlying high-dimension data, they don't preserve local or global structure & are misleading. They're also arbitrary." https://twitter.com/lpachter/status/1431325969411821572?s=21&t=HtzVmulBKba77ShXSQcKIQ
  - "My rule of my thumb, if the data has structure it should be immediately obvious. PCA then UMAP is a reasonable place to start. Never a good idea to fiddle parameters until you find what you're looking for."
  - "Well, I’ve used a lot of umaps in my day, but scanpy has really convenient plotting tools for heatmaps, dotplots,  violin plots, and much more. https://scanpy-tutorials.readthedocs.io/en/latest/plotting/core.html"
- Still don't understand ATAC. It seems to be measuring something different from just genes

## Random ideas
- Since there is good knowledge of how different bone marrow stem cells differentiate, seems important to incorporate. 
- I have a hunch that most ATAC data will be correlated with nothing because it's either non-coding or there was no transcription factor. 
- Dna and Rna expression could be passed through [Unirep](https://github.com/ElArkk/jax-unirep) as a form of dimensionality reduction. This was actually performed in last year's competition: https://github.com/openproblems-bio/neurips2021_multimodal_topmethods 
- Someone notes that most of the donors are male, but one is female. Brainstorm other characteristics and features to extract
- Since we are working with count data (albeit normalized), I'm thinking poisson regression might lend a more powerful prior. It seem NBD modeling is often used, and would be more powerful since it can have separate variance: https://www.kaggle.com/competitions/open-problems-multimodal/discussion/346341 
- Make sure to account for promoter regions/transcription factors as genes
  - https://www.kaggle.com/competitions/open-problems-multimodal/discussion/349242
  - how to map names: https://www.kaggle.com/competitions/open-problems-multimodal/discussion/349559
  - Inputs: For the RNA counts, each row corresponds to a cell and each column to a gene. The column format for a gene is given by {EnsemblID}_{GeneName} where EnsemblID refers to the [Ensembl Gene ID](https://www.ebi.ac.uk/training/online/courses/ensembl-browsing-genomes/navigating-ensembl/investigating-a-gene/#:~:text=Ensembl%20gene%20IDs%20begin%20with,of%20species%20other%20than%20human) and GeneName to the gene name.

