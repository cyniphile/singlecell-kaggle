## Generation Notes
https://www.kaggle.com/competitions/open-problems-multimodal/data

- Three data modalities from two assays:
  - https://www.10xgenomics.com/products/single-cell-multiome-atac-plus-gene-expression
    - ATAC data - Assay of Transcription available chromatin
    - GEX data - gene expression (rna)
  - https://support.10xgenomics.com/permalink/getting-started-single-cell-gene-expression-with-feature-barcoding-technology
    - https://www.biolegend.com/en-gb/products/totalseq-b-human-universal-cocktail-v1dot0-20960
    - surface protein
	- Gene expression (rna)
- Taken at five time points over a 10 day period. (Is there any known biology about timing?)

SO there are two gene expression datasets?

*Task:*
For the Multiome samples: given chromatin accessibility, predict gene expression.
For the CITEseq samples: given gene expression, predict protein levels.


## Engineering Notes
- Data is big, and I don't have a GPU :-(. Might have to use cloud notebooks...or buy one (-___-)
  - 100hr/month Free compute with [Saturn Cloud](https://www.kaggle.com/competitions/open-problems-multimodal/discussion/346999#1909222)
    - Luke's: https://app.community.saturnenterprise.io/dash/o/community/user-details/
- At least see what is possible on a downsampled dataset. 
 
 - On visualization:
  - "It's time to stop making t-SNE & UMAP plots. In a new preprint w/ Tara Chari we show that while they display some correlation with the underlying high-dimension data, they don't preserve local or global structure & are misleading. They're also arbitrary." https://twitter.com/lpachter/status/1431325969411821572?s=21&t=HtzVmulBKba77ShXSQcKIQ
  - "My rule of my thumb, if the data has structure it should be immediately obvious. PCA then UMAP is a reasonable place to start. Never a good idea to fiddle parameters until you find what you're looking for."
  - "Well, I’ve used a lot of umaps in my day, but scanpy has really convenient plotting tools for heatmaps, dotplots,  violin plots, and much more. https://scanpy-tutorials.readthedocs.io/en/latest/plotting/core.html"


