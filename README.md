# LQ-asym

This branch is the updated version of the main branch and now supports the new LQ samples. I also made a few more changes and fixes that I missed in the main branch that I pushed at the end of my summer student contract. Some more details about the project itself are there. At some point I may merge this branch with that one, but my CERN account deactivates very soon so I likely won't have time.


## Prerequisites

### Project Setup
This project should be set up in the following way: first, clone this branch of the repository to your EOS on LxPlus as well as somewhere local on your machine. For each place you clone it, create a python virtual environment with `python3 -m venv .venv` in the root of the project, and install all of the required libraries using the `requirements.txt` file.

The ntuples should be copied from wherever they are in ATLAS's EOS to somewhere in yours. I trust that you either know where this is yourself or can contact someone who does. Additionally, since the total size is ~50/60GB, it is feasible to copy them also to your local machine, if you'd like.

### HTCondor
A few of these steps require HTCondor usage, and currently there isn't support for running jobs from EOS. Because of this, create an identical folder in your AFS called `lq-asym`, then copy (or move, if you'd like) the `jobs` and `.condor` folders from the project into this new AFS folder. Any time I reference running HTCondor jobs, then we are considering these directories from within your AFS, not the EOS ones.

### Python
The default pip cache directory is in your AFS, and the cache will exceed the maximum 2GB you are allotted very quickly most likely. So, set the pip cache directory to somewhere in your EOS like so:

```
export PIP_CACHE_DIR=<some-path-in-your-EOS>
```

You could put this in the `.bashrc` to have it run every time you ssh into lxplus.


### TRExFitter

In the main branch, I complained about how hard it is to use TRExFitter anywhere but lxplus. This isn't entirely true; getting a Docker container set up on my local machine wasn't actually all that bad. In principle, then, this entire pipeline could be run on your local machine, but would just take a while without HTCondor as you'd just run the scripts themselves without any parallelization.

It is still more convenient on lxplus, since all you have to do is:

```
setupATLAS
asetup StatAnalysis,0.4.0
```

or whatever version is more recent. This one still works fine, so there's no real need to change it.

Also, the replacement files contain a lot of information about the samples and selection criteria and whatnot. That is why often in some seeminly unrelated steps I say to edit a TRExFitter replacement file.




## Pre-Fit Yields

This step is in principle not necessary, but it is good to get an idea of the distribution of basic kinematic/other variables for our samples. First, in the `trex-fitter` directory, open the `replacements/replacement.txt` file and at the top, edit the line `XXX_NtuplePaths` to point to your copy of the ntuples.

From there, check out the file `configs/pre-fit.config` if you want to see what it looks like, then run (still from within the `trex-fitter` directory)

```
trex-fitter nwd configs/pre-fit.config
```

This will go through all the ntuples, so it'll take some time. Alternatively you could go through the small ntuple step first, then point the `XXX_NtuplePaths` variable to point there, and it'll take a significantly shorter time at the cost of better statistical significance.

After it finishes running, the plots of interest will be in `outputs/pre-fit`.



## Small Ntuples

The raw ntuples are large, so for some parts of the analysis we would like to have ntuples that are as trimmed down as possible. We do this by applying a selection criteria to all the ntuples and saving new, "small" ntuples that contain the events that survive. For this, we use HTCondor.

Back in your AFS, navigate to the `jobs` folder that you copied earlier. In here is a script called `get_names.py` and a text file called `names.txt` If no new LQ samples have been added yet, then don't worry about this, but if there have been new ones added and you've copied them to your EOS, then open `get_names.py` and change the path to your own EOS ntuple storage, then run the script. It just looks at the ntuple directory and grabs all the names of the files and places then in `names.txt`.

With this, go into the `produce_small` folder and open both files and change paths to point to your copies of stuff; this should be straightforward. Then, just run

```
condor_submit condor_submit.sub
```

You should see ~875 jobs submitted, more if there have been more LQ samples produced. It shouldn't take too long to run. The files will be output back to your EOS version of the project, in the `friend_ntuples/output/small` folder.


## Data Processing

Next, we need to transform the small ntuples into numpy files so that Pytorch can read them. All that is needed to be done is to change the `XXX_NtuplePaths` variable inside the `replacement_small.txt` file within the TRExFitter configs to point to your version of the small ntuples. While you're there, go ahead and change the path to the friend ntuples to point to your own. All that this should entail is changing my user to yours in the path.

With this, just run

```
python3 data_processing/main.py
```

The numpy files will be put in `data_processing/output`.


## Machine Learning

We now copy these small ntuples to the version of this project on the local machine. Place them in the same directory, namely, `data_processing/output`.

The `requirements.txt` file does not contain pytorch, since we don't want it on lxplus as it'll take absolutely ages to install. So, on your local machine, ensure you're in your virtual environment and run `python3 -m pip install torch`. Or, if you are training on lxplus, you can just install pytorch there.

I edited none of the machine learning code, so there is nothing really to change here. If you go to `ml/configs`, you can browse around for the different models that are available, or make your own. I did a ResNet-6, and for this I typed:

```
python3 ./ml/train.py resnets/resnet-6
```

where I omit the file extension.

This also uses WandB, so set up an account there.

### Note

If you are going to be running multiple iterations (which you almost surely are), make sure that you go to the WandB website and either rename, move, or delete any previous runs of the same model. When doing the friend ntuples, the code searches by name for a matching model/run name, but it doesn't take into account time of creation. It may grab a previous model. Renaming/moving/deleting previous runs ensures there is only one run with that name so there is no ambiguity as to which one it uses.


## Friend Ntuples

To get the output of the model on our samples, we create friend ntuples, which are new ntuples that have only one branch: the output of the model which classifies the probability of an event being signal (LQ). This should also be done on the local machine. If you did the training on lxplus, then do this step also on lxplus, to keep it consistent.

We first need to copy all of the small ntuples to our local machine as well, and just as with the numpy files, place them in the same directory, namely `friend_ntuples/output/small`. All of the paths at this point should be relative, meaning that this is all that should have to be done. Now we just run

```
python3 friend_ntuples/produce_friend.py
```

The outputted files will be placed in `friend_ntuples/output/friend/<model-name>`.

### Note

There is a comment within the `friend-config.yaml` file that requires some attention. Since all these files are relative (rather, they use the `~`) there is no reason to go and change anything within it, but if you need to, take care in noting that the `source_base_dir`, if it contains a trailing forward slash, will break the code for whatever reason. Additionally, the `target_base_dir` _must_ contain a trailing slash or else the code will break. This is something that I could have gone in and fixed, but the code is really just a mess and I didn't want to have to deal with that.




## Probabilities and Statistical Uncertainties

Copy the friend ntuples back to lxplus in the same directory. In the file `trex-fitter/replacements/replacement.txt`, edit `XXX_FriendPaths` to point (as an absolute path) to your friend ntuples. Then, from within the `trex-fitter` folder run

```
trex-fitter nwdfr configs/probs.config
```

and/or

```
trex-fitter nwdfr configs/probs_no-cut.config
```

The former will cut out events whose probability of being LQ is below 0.5, and the latter keeps every event (but still only those within the signal region). I have this setup this way since we currently have so few LQ samples that even though the model is decent, there are just not enough LQ events and relatively too many background events that we can't really see anything in the output plots. 

For a measure of the statistical uncertainty, only consider output from second one (the one without the cut), since that has all of the events and thus the best measure of uncertainty.

The files are output to `outputs/probs` and `outputs/probs_no-cut` respectively. The statistical uncertainty information is found in `outputs/probs_no-cut/NormFactor.(png/pdf)`.



## Next Steps

Assuming this pipeline works at all, I have been having difficulties with cut criteria and training and whatnot. It also may be beneficial at some point or another to again try training with those medium ntuples that I did before, just for more data.

In principle, we would have similar `p` tags for all the ntuples. The backgrounds being different from the LQ samples led to the removal of a few important variables used in cut criteria, and perhaps other general differences between the two are causing some issues. I believe this is in the process of being resolved in one way or another.

Lastly, since this entire release of ntuples is _very_ different from the previous release, there are still a number of issues with all the other missing variables. Perhaps, since this release's ntuples have been out for a while, it is possible to find someone to ask if there have been TRExFitter replacement/configuration files made (specifically within the Tau+X group, if possible) for this release. I know that I was given a link on the Tau+X eos for some of these a while back, but I believe they were for the previous ntuple release.

Once all of this is sorted out (which I can't imagine will be anytime soon, unfortunately), then time can be spent fine-tuning the machine learning model and stuff like that.
