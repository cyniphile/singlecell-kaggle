## 10/7

Working on getting that baseline model out. Used `TruncatedSVD` to reduce to 64 cols, and submitted my first full prediction. Still low on the totem but not terrible like before.

Next steps: 
- Clean up code
  - set up nice pipeline
  - set up cross validation
- Do ridge regression
- Might want to just research it too
- Regress on correlation vs MSE
- Try adding day as a feature (scalar)

## 10/6

Spent yesterday/this morning working on screenplay stuff. Today get some basic EDA done on the multiome data, get a better feel for handling `.npz` and sparse data, then do a downsampled linear regression. TBF, it probably makes sense to do some type of PCA from the get-go because, c'mon 200k columns??!! Goal, resubmit with a full two-task prediction. 

Wasted a good amount of time on figuring out how to just read the column names of an `.hd5` file. It turned out to be an series of arcane (and idiotic) tricks that baffled even Jason (neither of us had any idea what `list[()]` means).

In the end, all the columns have different labels across all data. There might be a way to parse them into each other, but will take extra work.

- TODO: How to merge with metadata? Especially "Day" 

## 10/5

Had a big boo-boo: accidentally canceled my job after running for almost 5 hours. I don't think I'm going to restart it just yet, though I do eventually need a baseline. Gonna stick with some downsampled version just to get a higher score on the LB and get a "my laptop" baseline. Should be able to train something with 10k rows for about 5.5 hours using 3 cores of my laptop.

I also didn't realize the input data was so wide (over 200k cols!) so really I need to take a break, do proper EDA on this dataset, and come back. I'm rushing!


## 10/4

Goal: Make predictions for Multiome, possibly using sparse matrices.

Got saturn cloud setup. Jupyter Lab in a browser is annoyingly 80% baked, so need to connect to VSCode. Saturn cloud also seems to have poor upload bandwidth, so I'm worried about getting models and predictions on off. https://saturncloud.io/docs/using-saturn-cloud/ide_ssh/ 

Also got sparse representations of all data (esp multiome) created. 

Biggest worry by far is my (currently still training :-0) linear regression model for multiome. It's been training locally for almost 6 hours, and on saturn cloud with 32 CPUs for almost 3 hours (!!) and still going (this without fully memory usage/swapping). I'm annoyed with myself for breaking my own rule of "always train a small model first", but I was thrown off by a sorta fast (40m) train time for the cite-seq data. Got sloppy and didn't do the simple math beforehand: the gene output is 23k columns wide, while the cite-seq protein output was only 140 columns wide. So that's ~200 times as wide, plus its 106k vs 71k rows long so 200 * 1.5 * 41 min / 32 cores = roughly 5.5 hours expected time, so SHOULD be coming to a close around 11pm tonight. 

Dowmsampling seems like it's gonna be necessary up until maybe the last minute of the competition, because I'm not gonna be able to afford to train at this rate for long.



## 10/3
~~Goal: make simple predictions for CITE-seq assay. For today I'm not gonna go much deeper into vector regression theory and just use basic sklearn methods.~~

Not spent much time on this so far other than lightly following discussions on Kaggle, but now getting some free time to focus/build.

My first task is to review vector-output regression methods (which I've not used before) as well as the competition schema. I'd like to make a simple set of predictions just to get a baseline on the LB and see how expensive predictions will end up being in the simplest case. Today just do one of the assays, probably CITE-Seq.

Got a halfway decent submission working for Cite-seq only.  Took 40m 24s to train a linear regression with sklearn. Lots of swapping. 

Now looking if sparse matrices could help with compute for _much_ larger "Multi" data (which is apparently sparse). See: https://www.kaggle.com/code/sbunzini/reduce-memory-usage-by-95-with-sparse-matrices#!!-Update-!!

