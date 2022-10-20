# 10/19

Spent morning working on movie stuff and meeting with a friend. Now finishing up working out the kinks with making prefect work with dask. There is some async error going on. Goal for today to get experiment tracking with MLFlow and some automatic hyper-param optimization for smaller data. 

Ran into some weird bugs with the python debugger using jupyter notebooks with the `DaskTaskRunner`. Switching to a scripts as notebooks are proving too unwieldy in this more mature phase. 

# 10/18

Didn't quite finish implementing data pipelining yesterday (bad, spanks, spent a little too long doing research on data tools, but damn is it a complex world. [This site](https://mymlops.com/) was very helpful in wrapping my head around it). Did get some initial work done in Prefect which actually seems great so far. 

Today want to finish up the porting, get MLFlow setup as well, and test how well it all works on kaggle/saturn cloud. May need to set up [Dask integration](https://docs.prefect.io/tutorials/dask-ray-task-runners/) to get proper parallelization

EOD: Finished up porting to prefect (mostly, at least was able to delete various ugly notebooks). 

# 10/17

I'm feeling like the next step is to set up local background hyperparam optimization and tracking local cv with submission more automatically. Again, papermill will be good at this.

I think another goal is to try using previous-day output as an input. Given the size of the data, using huge, all-data models is not really feasible anyway, and I'm getting decent scores using only 25k rows (roughly 25-35% of data). It seems like scores were asymptoting as I added more rows of training data. Also, I didn't properly randomize data (was just taking data from the top of the file) so maybe there was even some ordering I was not taking into account for. I'm guessing using more data + hyperparam optimization would get me into the .81 score range (.802 now). 

When using previous day input, I think the question is whether to use two models in ensemble, or train one big model. TODO: Probably want to start with the two model approach to see if the, say, Day 1 -> Day 2 protein model does anything useful on its own! Remember there is no row-wise mapping between targets, but overall distributions might change

*Review of pipeline tools*
Looking at documentation of all these dang tools for hours led me to 
choose [ZenML](https://github.com/zenml-io/zenml) as a pipelining tool.
- It seems decently popular. 2.4k stars on GH
- The docs make the most sense quickest. 
- It integrates easily with MLFlow
- It supports caching.
- It's oss, fairly light-weight.
However, I found on trying to install it only works with python 3.9 (this project is 3.10). So I went with [Prefect](https://docs.prefect.io/tutorials/first-steps/) which generally seemed to be even more liked. 

[MLFlow](https://github.com/mlflow/mlflow/) seems standard, hugely popular, and not too hard to integrate so gonna use that for experiment tracking.  



# 10/15 - 10/16
Didn't make much progress on setting up a good pipelining tool last friday. Tried building something, but realized there _has_ to be something out there, so doing research on that (so far `dagster` looking promising.) 

Also did some review of RBF kernel. It's complex, and not something to easily pick up expert-level theory (would probably take a couple full days, maybe more if I were to really learn Lagrangians as I've been wanting to do for years anyway.) I think the main thing to remember is that the `gamma` param is a sort of "number of model parameters" god param, so bigger gamma means more params/less "regularization". See: https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html


# 10/14
Continuing with this idea: I started using `papermill` (which I've been wanting to use for awhile). I got it set up to run a quick "unit test" of a notebook, which has been really useful. Would be good to parameterize 'my' notebooks further (also using `scrapbook` for outputs, and the kaggle api for processing). I also want to save intermediately processed data, (like pca) more automatically and idempotently. 

# 10/13
Should really do this every day, but catching up now. Ran a couple models: 1) a larger CITE model and 2) and the same RBF model on multi data to get a stronger benchmark on the LB. Also tooled a bit on making the notebook model more technology agnostic, mostly because multi really requires using sparse matrix inputs. 

Improved model score to 0.802 (99% of current winning score), but not done anything that interesting so far. Clearly there are non-linearities as RBF has improved everything over linear modeling. Not tuned hyperparams at all yet either.

# 10/12

Probably biggest blocker to making another submission with the RBF kernel is local CV. Only problem is CV is very expensive, and it's hard/expensive to even train one model. The public kaggle notebooks seem to just downsample aggressively, and the RBF solution from last year did this as well. 

Paired with @jasotron today and did a lot of code refactoring. 

Today need to 1) merge his branch, 2) finalize a system to keep track of performance locally (submit via api and centralize experiment data)
# 10/11

Today I'd like to get one of the winners from last year working. Their approach was simple-ish, and was for CITE data, which is both easier to work with and more important for scoring. Code here: https://github.com/openproblems-bio/neurips2021_multimodal_topmethods/tree/main/src/predict_modality/methods/Guanlab-dengkw/run

Important note that I'm doing a lot of research to essentially save on compute. Can't afford to try everything.

I'll submit with existing simple linear model for MULTI.
To review: 
    - 31:00 GEX->ADT. Row-wise normalized. Ride regression with RBF kernel. Kernel was expensive, with an ensemble with smaller regressions to save on memory. 

    - Q: Is GEX->ADT really the same as CITE? A: Yes!
    - Q: Double check forums about trying this. 
      - A: Only the Russians seem aware of this. I'm not sure what their deal is because they seem very active but also rather ineffective, with a high variance of usefulness to their many (many) posts. 
      - This notebook: https://www.kaggle.com/code/xiafire/msci-multiome-5-steps-x-5-folds-25-models Pretty heavily copied but no comments and not that many upvotes. No note of performance.

Got a basic RBF kernel model working, but not on full dataset. I want to look into a slightly more automated data pipeline/experiment management system, maybe viash. Also really need to get local cv working; it's not a good step to skip.

----------------------
# 10/10

Finished comprehensive review of discussions, saving a bunch of tabs to review. 
Some key points:

- I'd slightly misunderstood the evaluation metrics. In fact, we take correlation *per cell* and then average across cells (not correlation per cell-gene). Affects loss functions as well as data weight (CITE is higher weighted).
- The actual test for MULTI only uses 30% of the test data (weird) so can TODO: downsample test data accordingly: https://www.kaggle.com/code/ambrosm/msci-multiome-quickstart?scriptVersionId=103802624&cellId=14
- Some other things were confirmed/discussed
  - LB will be prone to shakeup because of the time structure of data. 
  - should do PCA on train+test input
  - some discussions of how many dims to use (>128 wasn't helpful) and what models (seems MLPs are slightly favored over tree methods).
- I think I've underestimated (as usual) the amount of work needed to do well here. Get ready to turn up the intensity.

Still makes sense to continue review, especially winners of last years comp but with a timeseries flavor. Keep more careful notes to stay focused/sharp.

### Review of strongest EDA notebook:
https://www.kaggle.com/code/ambrosm/msci-eda-which-makes-sense/notebook#EDA-for-the-Multimodal-Single-Cell-Integration-Competition
  - CITE
    - TODO: Should drop all-zero columns more carefully; 449 in CITE!
    - Some features seem poisson/nbd distributed, some not (see grid of hist of selected features)
    - Did an interesting plot of first two SVD components for all subsets of data.  https://www.kaggle.com/code/ambrosm/msci-eda-which-makes-sense?scriptVersionId=105637527&cellId=28
    - Grouped K-Fold leads to small data-per-feature. Overfitting is a big risk.
    - Protein outputs have diverse shapes. One-size-fits all modeling may not be best. https://www.kaggle.com/code/ambrosm/msci-eda-which-makes-sense?scriptVersionId=105637527&cellId=31
  - MULTI
    - This cell on experiment batch effects is important. Should account for this: https://www.kaggle.com/code/ambrosm/msci-eda-which-makes-sense?scriptVersionId=105637527&cellId=44
    - Should add a Male/Female feature: 13176 = Female, 31800, 32606, 27678 = male.

  Top public notebook: https://www.kaggle.com/code/pourchot/multiome-with-keras-ensemble/notebook

  - Fairly standard MLP
    - predicts reduced features, then does inverse PCA at the end
      - Thus, it does not actually optimize on correlation score, unlike some other top (but slightly inferior) notebooks, e.g. https://www.kaggle.com/code/ambrosm/msci-citeseq-keras-quickstart
    - ensembles with a few other public Notebooks

### Uses many of the tricks I've thought of and is well explained: https://www.kaggle.com/code/ambrosm/msci-citeseq-keras-quickstart

### Review of last year's competition: https://drive.google.com/file/d/1aQss-KyfYlzdrBQcH5joiXMlTwpG5gdf/view
- Detail: https://github.com/openproblems-bio/neurips2021_multimodal_topmethods

  - 5:30 batch effects
  - 12:09 Last year winner ADT -> GEX, GEX2ATAC
    - Pretty standard model with reverse pca outputs 
      - Two catboost, one MLP, one ??"Resnet style" MLP (skip-connection)
      - Ensembled
  - **These two are most important**
    - 21:00 ATAC->GEX An "all in one" encoder style NN with dropout on input. TODO: Important to look at "differentially accessible peaks" (??). Cell type didn't help. Code: https://github.com/openproblems-bio/neurips2021_multimodal_topmethods/tree/main/src/predict_modality/methods/cajal
    - 31:00 GEX->ADT. Row-wise normalized. Ride regression with RBF kernel. Kernel was expensive, with an ensemble with smaller regressions to save on memory. 
      - code: https://github.com/openproblems-bio/neurips2021_multimodal_topmethods/tree/main/src/predict_modality/methods/Guanlab-dengkw/run
  - 41:00 Overall winner across modalities: Used GNNs. Used pathways from molecular signature dataset to connect. Calculated batch level features. Did an ablation study. Most complex, but would have been bette to use different simple models. Graph convolution?? Code: https://github.com/openproblems-bio/neurips2021_multimodal_topmethods/tree/main/src/predict_modality/methods/DANCE/train
  - 53:00 Not as applicable to current competition and stopped watching

----------------------
# 10/8

I think now it's time to go back and do a more through review of public notebooks on kaggle. 

Additionally, it will be important to test a strategy that more intelligently takes time (`day`) information into account. So far I'm thinking that (where possible) I'll just use the previous day's output as an input to the next model. This means separate models for data with/without previous day's output available. Unfortunately this also means this model will not help in the public LB, but should help in the private one, which (TODO: is this true?) is the measure of winners.

Finally, remember we have multimodal data always. Thus we can always use multi and cite inputs together, which I think could make sense. True, we can't match per cell, but I bet there's a way to map per individual from general concentrations of, say ATAC to Prot and vice versa. Imagining a model that per individual predicts some simple boost of prot based on availability.

----------------------
# 10/7

Working on getting that baseline model out. Used `TruncatedSVD` to reduce to 64 cols, and submitted my first full prediction. Still low on the totem but not terrible like before.

TODO: Next steps: 
- Clean up code
  - set up nice pipeline
  - set up cross validation
- Do ridge regression
- Might want to just research what others have done on Kaggle already too. Really not a ton of point in redoing work. 
- Optimize on correlation vs MSE
- Try adding day as a feature (scalar)

----------------------
# 10/6

Spent yesterday/this morning working on screenplay stuff. Today get some basic EDA done on the multiome data, get a better feel for handling `.npz` and sparse data, then do a downsampled linear regression. TBF, it probably makes sense to do some type of PCA from the get-go because, c'mon 200k columns??!! Goal, resubmit with a full two-task prediction. 

Wasted a good amount of time on figuring out how to just read the column names of an `.hd5` file. It turned out to be an series of arcane (and idiotic) tricks that baffled even Jason (neither of us had any idea what `list[()]` means).

In the end, all the columns have different labels across all data. There might be a way to parse them into each other, but will take extra work.

- TODO: How to merge with metadata? Especially "Day" 

----------------------
# 10/5

Had a big boo-boo: accidentally canceled my job after running for almost 5 hours. I don't think I'm going to restart it just yet, though I do eventually need a baseline. Gonna stick with some downsampled version just to get a higher score on the LB and get a "my laptop" baseline. Should be able to train something with 10k rows for about 5.5 hours using 3 cores of my laptop.

I also didn't realize the input data was so wide (over 200k cols!) so really I need to take a break, do proper EDA on this dataset, and come back. I'm rushing!


----------------------
# 10/4

Goal: Make predictions for Multiome, possibly using sparse matrices.

Got saturn cloud setup. Jupyter Lab in a browser is annoyingly 80% baked, so need to connect to VSCode. Saturn cloud also seems to have poor upload bandwidth, so I'm worried about getting models and predictions on off. https://saturncloud.io/docs/using-saturn-cloud/ide_ssh/ 

Also got sparse representations of all data (esp multiome) created. 

Biggest worry by far is my (currently still training :-0) linear regression model for multiome. It's been training locally for almost 6 hours, and on saturn cloud with 32 CPUs for almost 3 hours (!!) and still going (this without fully memory usage/swapping). I'm annoyed with myself for breaking my own rule of "always train a small model first", but I was thrown off by a sorta fast (40m) train time for the cite-seq data. Got sloppy and didn't do the simple math beforehand: the gene output is 23k columns wide, while the cite-seq protein output was only 140 columns wide. So that's ~200 times as wide, plus its 106k vs 71k rows long so 200 * 1.5 * 41 min / 32 cores = roughly 5.5 hours expected time, so SHOULD be coming to a close around 11pm tonight. 

Dowmsampling seems like it's gonna be necessary up until maybe the last minute of the competition, because I'm not gonna be able to afford to train at this rate for long.



----------------------
# 10/3
~~Goal: make simple predictions for CITE-seq assay. For today I'm not gonna go much deeper into vector regression theory and just use basic sklearn methods.~~

Not spent much time on this so far other than lightly following discussions on Kaggle, but now getting some free time to focus/build.

My first task is to review vector-output regression methods (which I've not used before) as well as the competition schema. I'd like to make a simple set of predictions just to get a baseline on the LB and see how expensive predictions will end up being in the simplest case. Today just do one of the assays, probably CITE-Seq.

Got a halfway decent submission working for Cite-seq only.  Took 40m 24s to train a linear regression with sklearn. Lots of swapping. 

Now looking if sparse matrices could help with compute for _much_ larger "Multi" data (which is apparently sparse). See: https://www.kaggle.com/code/sbunzini/reduce-memory-usage-by-95-with-sparse-matrices#!!-Update-!!

