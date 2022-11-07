# 11/7

# 11/4

Need to address this cv/lb problem. A few ideas: 1) almost certainly need to do random sampling now 2) Probably need to do grouped k-fold by day 3) Should also just do a quick double check to make sure there's not some other bug, as it is true CV so far is usually really high. Did some checking, CV seems pretty legit as is.



# 11/3

Did some more memory profiling. Using `SequentialTaskRunner` decreased peak memory by 22%. I was still seeing some large objects from `.load_data()` in the peak report. I could be reading the graph incorrectly, but seems almost like large objects aren't being de-allocated properly, so doing some manual `del df; gc.collect()` to see if that helps. Before (and this didn't help) I had been doing just `del` without `gc.collect()` but who knows (I think the technique is more for notebooks). Aaand yes, the python gc DOES work by itself (no change at all in peak mem usage). 



That said I think I need more reduction of peak mem.

One option, switch to SVR, but I'd rather purse later as I seem to be getting good models as is: https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_ridge_regression.html#:~:text=They%20differ%20in%20the%20loss,than%20SVR%20at%20prediction%2Dtime. I'm also not sure if SVR, though faster for larger data, uses less memory. 

I'm gonna dive into Dask more deeply, because out of core memory stuff is supposed to be its key value add.

I'm being reminded that I've not really done a proper learning curve yet. I so have SOME data from past submission where, (with 128 input/output dims) I got a slightly better score going from 25k row (.801 -> .802). Gonna do a submission with what I CAN do (50k rows multi/cite) just to see how things work.

Another way to solve these memory issues is to loop over each output and train a model for each. Sklearn seems to be poor at optimizing this. 

Saturn cloud was having issues submitting to kaggle, so had to download, THEN submit. Aaaand, terrible score. Unsure how I'm getting such a good CV score though, very high on multi and cite. Is something wrong with my pipeline? Could be that I've not set up random sampling and that's biasing things.

Seems like there's a bug in my cv locally, as correlation is not so high with lb score.

#11/2

Several goals for today, the biggest is to reduce memory usage. To do that I'm going to need to learn how to use memray, probably use dask. 

Spend a long time on a wild good chase trying to get dask's visualizer to work, culminating with this bug report: https://github.com/dask/distributed/issues/7244. Re-learned to use `ipdb.set_trace()` the good old fashioned way when vscode's debugger wasn't working. Big waste of time in general, but so it goes. 

Did some memray profiling with memray. I guess not surprising, but Dask for prefect increases peak memory usage (doing stuff in parallel doi)

# 11/1

Damn, lots of hardware issues last night after all. Many jobs suddenly die. I _think_ it's because I'm running out of memory, as it happens when lots of jobs are running and also tends to happen for multi much more easily. I don't know for sure though. Might need to enable swap space, as it seems for the most part memory is not used up.

I might just need to do a more serious profile of memory usage. Memory use suddenly exploding when fitting RBF. Might be better to use an SVM as they scale better? Or another model entirely that parallelizes better?

Ended up screwing around fairly inefficiently with different task runners for prefect, but I think the real problem is the memory consumption of the core task logic. I found memray for memory optimization did some more research on dask. I think doing a solid memory profile of my code + daskifying some key parts of it will help greatly.

I did managed to get some slow/inefficient cv in and both CITE and Multi models seem quite promising. 

# 10/31

While I still have some free saturn cloud hours I'm going to figure out this parallelism problem. Transitioning to a cloud infra is a little more work than I was (foolishly) expecting, but hope it won't take more than today. 

First, caching does, in fact, seem to work. Jobs are getting marked as "cached" and run for 0s

Might need to look into understanding prefect deployments and agents some more. Ok, so a "deployment" is a flow that's packed up to be run anywhere (as defined in a yaml file). Once you've converted a flow to a deployment, then you  can submit it to "agents" which are basically servers configured to listen for work. It seems pretty simple! There seems to be some "concurrency limit", but otherwise not sure how work is balanced across resources. I think this is a bit more the role of dask. Only problem is that my caching decorator breaks in this setting.

Ya know, re-looking at the results of my weekend experiments, I really am over-engineering infra at this point. My infra works well! So what if I'm sorta messily duplicating notebooks with different params at this point, mlflow makes that ok! I only have one box to work with anyway, so I can manually tune how much work I want it to do. 

There seems be an issue where ipykernel on saturn cloud "dies" after awhile, but the work seems to continue?

Ugh. Found a big bug where I was loading the wrong data. Might have over engineered how I was getting the data. Too many layers of manually matching of "train_test" "test_target". 

Retraining some models and (surprise) they are taking a lot longer with the full train set (which is much bigger). 

Thinking deep learning might be a good next step to take the time structure into account. 


#  10/29 - 10/30

Seems like tasks aren't running asynchronously. Go back to trying out https://docs.prefect.io/tutorials/execution/
Also need to fix caching for non `--full_submission` jobs. 

The uncertainty of `failed` or `running` in prefect now seems rather serious as I'm unsure whether to restart certain jobs. https://github.com/PrefectHQ/prefect/issues/7239

Also, caching seems to not be working, might have to use "local save" flag. 

I just manually ran a bunch of jobs with various settings in notebooks on the server. Felt like I wasn't maxing out my infra as well as I could, but feels like a decent system so far. Getting good logs, good model observability. One thing, need to have train scores be the same as the evaluation metric (the weird correlation). As is I'm just getting r2 scores which is an ok proxy but hard to tell if I'm over/under fitting. 

So far, some pretty exciting results according to my CV! Definitely getting there with "loose" (overfit) models, so we'll see if the results hold up on the LB. Sometimes overfitting is the way though...

Finally, showing the payoff on the infra I've put together. I think I could do it all again in 20% the time for my next project, and it's leading me to a really robust, trustworthy experimentation process.


# 10/28

Spent the morning implementing a function that, given mlflow experiments, will retrain, predict, and submit. This really show the power of mlflow, in that I can screw around with experiments, go back and find which ones did the best, and just paste those IDs into this function to submit. Seems soo easy and fault tolerant. I'm reflecting on how much time I've spent on pipelines and infra. Was it worth it? I spent two weeks researching, testing, and getting a truly production-ready pipeline setup. I honestly don't think this is that bad, as long as my goals for this project are NOT just to get the highest place on kaggle. I genuinely don't think so. I could have copied the highest scoring public notebook and instantly get 300ths place, and probably fuck around with hyperparams and feature engineering to squeeze out a top 100. That not only doesn't seem fun, it doesn't seem productive.  Now after a long pause in data sci, I feel like tooling has become modern, and it's important to update my ways. Yes, prefect had some issues, but it also seems to have great momentum and response, so it's worth betting on it continuing to exist. One thing I still need to test is how easy it is to use all this in the cloud. I think this will be my afternoon. Could I really be done done done with infra???. Got mlflow working as an accessible server running on saturn cloud, nice! Now I can submit experiments with confidence and abandon.

# 10/27

Spent the morning reading papers that make use of pseudotime methods, and found some tools, possible best practices:

Spent the afternoon doing more work on my pipelines to make my flow caching
function into a reusable decorator. Did more research into mlflow usage, capabilities and best practices, because I really should be submitting to kaggle based on experimental results that are logged in MLFlow.

# 10/26

Iteration is slow testing these larger jobs. I think in general I need to spend more time setting up test cases that don't take too long to run. If an iteration is a minute or five, that not only slows development a lot but also causes focus-loss because you have to either wait or context switch during the training. Hard to overcome the overhead of and complexity of writing a "small" test case when you aren't yet sure if it will pay off, but I think prob need to make a rule for myself: if a test takes > 30s to run, rewrite.

Decided to dig deeper into why wrapping a certain function in `@task` was causing a huge performance regression. I probably wouldn't do this if I was at a real job, but maybe I would. It's important to have some intuition for why things break. Then again, it's been too long since I actually iterated on real predictions. Filed a bug report, and also discovered a solution to some of these issues: tasks and flows should not return numpy objects. Making a task return a dataframe instead of a numpy object eliminated performance issues cause by wrapping in `@task`

EOD, I think I finally have pipelines done for now. Starting research into how to use pseudotime algorithms as I think properly modeling temporal data will be the biggest advantage here. So far I've only seen people treating Day as a feature and maybe doing `GroupedKFold(day)`

# 10/25

Ok goal is to as quickly as possible (as possible) set up a simple caching function for flow outputs. Basically hyperparams, data, model. I think the biggest weak spot here is the "data". Need to rework slightly as an input..probably just function `__name__`.

Got a solution working, though it took a lot longer than I hoped because apparently `hash` is not consistent across python runs. 

Was also running into some serious slowness with writing output to csv, so switched to arrow/feather which is supposed to be faster for I/O.

I also did more experiments with de-activating or removing all Prefect functionality to see if it was causing slowdowns. With prefect completely remove runtime was about the same as with it. The only exception, which I've confirmed again, is the `format_submission` task.

# 10/24

Still trying to figure out what do about pipelining. It's brought some nice benefits, such as how it stores metadata and puts it all in a nice web ui. But it's also caused performance issues which I can't afford. I'm also afraid it's become more of a distraction than an aid (though I can chalk of some of the dev time to amortizable learning and setup, it's getting to be too much). 

Today I'm gonna do a blitz to try out dagster. If I have any problems, depending on how things go I'll either just go back to where I am now (using prefect) and manually implement some things, or continue with dagster and manually implement some things. Ok a few hours in and dagster is proving pretty annoying to work with. The main problem is all functions need to be rewritten as `ops` which is much more wonky than the simple `task`s of prefect. For example, you can't just pass arguments in to an `op`, you have to pass a dict of parameters to the decorator. This feels crazy...everything has to be checked at runtime then? or you have to write a separate dataclass for each function? Some parts of dagster seem great, for example the asset / materialization api seems to be just what I need for caching model results. I even set up a nice pipeline for getting the data from kaggle an unzipping it. However not worth the price. 


# 10/22-10/23

Did some more research into alternative tools. Airflow just seems like a bad idea, as people complain it's not that great locally, and looking at the docs, it doesn't seem very incrementalist. People had complained that Dagster was higher cognitive overhead than prefect, but I took a look at the docs and they actually made a lot of sense. The tool also seems easy to adopt incrementally. Most importantly it SEEMS to support caching of `jobs` (their version of prefect's `flow`). However, it's listed as an "experimental" feature. 

Also, thinking about the `day` features, and how to best use them. Need to research ordinal features.

# 10/21

Refactoring workflow dag to cache more efficiently and merge and submit models more cleanly. Only problem is I'm running into some weird hanging behavior on the submission formatting that I've not had before. Gonna try uninstalling modin, and prefect-dask to ensure none of that bs is somehow getting in the way. This step used to take like 5min at most.

I also deleted some `del` statements (thinking maybe the gc wasn't doing it's job) gonna test if that was the problem. Python was using ~17g of memory before, now...same. I take it back gc...ur good.

Maybe it's because I put this function inside a prefect `task`? I took it out of the `task` wrapper and things went a lot faster. Maybe it's trying to cache or serialization too much?  This makes me worried about using prefect tasks. What is the true cost of wrapping a function in `task`?? Does this mean setting up lots of tasks and flows will be detrimental whenever you're on a single machine? Seems like it!

Also caching is not working the way I expect (same params run twice will use cached data if available). But this is not happening. After a long investigation, it's turns out you can only cache tasks, but not flow. WTF!! This makes no sense. My question of "[how to cache flows](https://github.com/PrefectHQ/prefect/issues/7288)" turned into a feature request. Easy caching or memoization was basically the main reason I turned to 3rd party workflow tools in the first place!

Side note: prefect seems not great at knowing when a flow/task has finished. Lots of tasks considered "running"



# 10/20

Ok bucko (jordan peterson voice), getting a bit sloppy on goals and follow through here. Today, hard deadline, get all experiment tracking set up, and do one experiment. MLFlow turned out to be quite easy to setup and nice to use. 

Some mlflow issues TODO:
```
2022/10/20 12:17:25 WARNING mlflow.sklearn: Training metrics will not be recorded because training labels were not specified. To automatically record training metrics, provide training labels as inputs to the model training function.
2022/10/20 12:17:25 WARNING mlflow.sklearn: Failed to infer model signature: the trained model does not specify a `predict` function, which is required in order to infer the signature
2022/10/20 12:17:25 WARNING mlflow.sklearn: Model was missing function: predict. Not logging python_function flavor!
```

The bigger issue was trying to use dask with prefect. Led to big slowdowns on the larger data. First, got the following warning:
```
/Users/luke/projects/singlecell-kaggle/.venv/lib/python3.10/site-packages/distributed/worker.py:2823: UserWarning: Large object of size 89.76 MiB detected in task graph:
  {'task': <prefect.tasks.Task object at 0x1333e28f0 ... ENABLED=True))}
Consider scattering large objects ahead of time
with client.scatter to reduce scheduler burden and
keep data on workers

    future = client.submit(func, big_data)    # bad

    big_future = client.scatter(big_data)     # good
    future = client.submit(func, big_future)  # good
  warnings.warn(
```

Also getting this random error sometimes:
```
2022/10/20 14:26:13 WARNING mlflow.sklearn.utils: RegressorMixin.score failed. The 'training_score' metric will not be recorded. Scoring error: shapes (666,666) and (667,4) not aligned: 666 (dim 1) != 667 (dim 0)
```
Seem to happen when I use the `run` command in vscode:

```
 source /Users/luke/projects/singlecell-kaggle/.venv/bin/activate
 /Users/luke/projects/singlecell-kaggle/.venv/bin/python /Users/luke/projects/singlecell-kaggle/code/rbf_pca_normed_input_output.py
```

...but not when I run the same script using `test.sh`. 

This only happens when is use the `DaskTaskRunner`. It also seems to have wildly unpredictable compute times. 

Given the problems I've been seeing with dask, gonna dump it for now and try `modin`. Modin seems immature. Lots of odd assertion and other errors littering logs even on successful runs. Also did not speed up things for my workflow (maybe not enough data manipulation yet?)
![The slow numbers for each `max_rows_train` are all modin](2022-10-20-19-00-00.png)


-----

Meanwhile, mlflow throws an error:
```
INVALID_PARAMETER_VALUE: Model registry functionality is unavailable; got unsupported URI './mlruns' for model registry data storage. Supported URI schemes are: ['postgresql', 'mysql', 'sqlite', 'mssql']. See https://www.mlflow.org/docs/latest/tracking.html#storage for how to run an MLflow server against one of the supported backend storage locations.
```
Turns out this only is relevant if you're using it to serve models (I'm not).


# 10/19

Spent morning working on movie stuff and meeting with a friend. Now finishing up working out the kinks with making prefect work with dask. There is some async error going on. Goal for today to get experiment tracking with MLFlow and some automatic hyper-param optimization for smaller data. 
 
Ran into some weird bugs with the python debugger using jupyter notebooks with the `DaskTaskRunner`. Switching to a scripts as notebooks are proving too unwieldy in this more mature phase. Then stayed up late into the researching experiment trackers as I'm getting cold feet on mlflow a bit. That said, it seems like the top choice, maybe other than WandB. ClearML also seems interesting. I'm going probably a bit too far into tool optimization, but hopefully I can not worry about tools for an ml project for a long time.

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

