
## 10/4

Goal: Make predictions for Multiome, possibly using sparse matrices.


## 10/3
~~Goal: make simple predictions for CITE-seq assay. For today I'm not gonna go much deeper into vector regression theory and just use basic sklearn methods.~~

Not spent much time on this so far other than lightly following discussions on Kaggle, but now getting some free time to focus/build.

My first task is to review vector-output regression methods (which I've not used before) as well as the competition schema. I'd like to make a simple set of predictions just to get a baseline on the LB and see how expensive predictions will end up being in the simplest case. Today just do one of the assays, probably CITE-Seq.

Got a halfway decent submission working for Cite-seq only.  Took 40m 24s to train a linear regression with sklearn. Lots of swapping. 

Now looking if sparse matrices could help with compute for _much_ larger "Multi" data (which is apparently sparse). See: https://www.kaggle.com/code/sbunzini/reduce-memory-usage-by-95-with-sparse-matrices#!!-Update-!!

