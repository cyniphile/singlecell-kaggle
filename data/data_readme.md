## Cite-seq 
 train_cite_inputs.h5 | features (rna genes) 70_988 * 22_050
 train_cite_targets.h5 | targets (protein genes) 70_988 * 140 cols
 test_cite_inputs.h5 | test features 48663 * 22050

## Multiome
 train_multi_targets.h5 | features 105942 * 228942
 train_multi_inputs.h5 | labels 105942 * 23418
 test_multi_inputs.h5 | test features 55935 * 228942

## Meta
 evaluation_ids.csv | A map from `cell_id` X `gene_id` -> `row_id`
 metadata.csv | Maps `cell_type`, `donor` and `technology` -> `cell_id`
 sample_submission.csv | `row_id` -> `count` (of expression)

## Error
 test_cite_inputs_day_2_donor_27678.h5 | Test set cells got replaced with training set cells. Can ignore?  See: `data_problem_fix.ipynb` 
 metadata_cite_day_2_donor_27678.csv  | same as above but for this day/donor

## Sparse Data
https://www.kaggle.com/datasets/fabiencrom/multimodal-single-cell-as-sparse-matrix

Each "xxx.h5" file is converted into two files:

- One "xxx_values.sparse" file that can be loaded with scipy.sparse.load_npz and contains all the values of the corresponding dataframe (i.e. the result of df.values in a sparse format)
- One "xxx_idxcol.npz" file that can be loaded with np.load and contains the values of the index and the columns of the corresponding dataframe (i.e the results of df.index and df.columns)

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

TODO: So there are two gene expression datasets??

