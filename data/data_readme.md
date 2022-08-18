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
- At least see what is possible on a downsampled dataset. 
 
