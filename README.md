# LQ-asym

This branch is the updated version of the main branch and now supports the new LQ samples. I also made a few more changes and fixes that I missed in the main branch that I pushed at the end of my summer student contract. Some more details about the project itself are there. At some point I may merge this branch with that one, but my CERN account deactivates very soon so I likely won't have time.


## Prerequisites

### Project Setup
This project should be set up in the following way: first, clone this branch of the repository to your EOS on LxPlus as well as somewhere local on your machine. For each place you clone it, create a python virtual environment with `python3 -m venv .venv` in the root of the project, and install all of the required libraries using the `requirements.txt` file.

The ntuples should be copied from wherever they are in ATLAS's EOS to somewhere in yours. I trust that you either know where this is yourself or can contact someone who does. Additionally, since the total size is ~50/60GB, it is feasible to copy them also to your local machine, if you'd like.

### HTCondor
A few of these steps require HTCondor usage, and currently there isn't support for running jobs from EOS. Because of this, create an identical folder in your AFS called `lq-asym`, then copy (or move, if you'd like) the `jobs` and `.condor` folders from the project into this new AFS folder. Any time I reference running HTCondor jobs, then we are considering these directories from within your AFS, not the EOS ones.

### Python
Pytorch installs a number of packages that are quite large. The default pip cache directory is in your AFS, and the cache will exceed the maximum 2GB you are allotted very quickly. So, set the pip cache directory to somewhere in your EOS like so:

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



## Data Processing / Root-To-Numpy

To do the training, we need numpy files as opposed to ROOT files, as pytorch has no idea what those are. In principle we would convert the raw ntuples into numpy files, but on my machine these were too big to be loaded into pytorch, so I made "medium" ntuples that are half the size. The script to do this is again back in the AFS jobs folder. Navigate to `root_to_numpy` and open the files in that directory and change paths as needed, then submit the jobs in the same way as for the small ntuples.

The medium ntuples will be output in the same place as the nominal ntuples. For instance, if you have the nominal ntuples saved in `<your_eos>/ntuples/nominal`, then the medium ones will be output to `<your_eos>/ntuples/medium`.

After producing the medium ntuples, you will need to edit the TRExFitter replacement file found in `trex-fitter/replacements/replacement-medium.txt`. This is an identical copy to the original replacement file, but you will need to change the `XXX_NtuplePaths` variable to point to the medium ntuples instead. This should just involve a change of the final bit in the path.

The processing of all the ntuples takes long enough that lxplus will kill it, so we need HTCondor again. As before navigate to the AFS `jobs` folder and go to the `root_to_numpy` folder. Edit the paths within the files accordingly and submit the job. This one isn't parallelized, so it may take some time.

The files are output to `data_processing/output` in the EOS version of the project.



## Machine Learning

We now copy these files to the version of this project on the local machine. Place them in the same directory, namely, `data_processing/output`.

The `requirements.txt` file does not contain pytorch, since we don't want it on lxplus as it'll take absolutely ages to install. So, on your local machine, ensure you're in your virtual environment and run `python3 -m pip install torch`.

I edited none of the machine learning code, so there is nothing really to change here. If you go to `ml/configs`, you can browse around for the different models that are available, or make your own. I did a ResNet-6, and for this I typed:

```
python3 ./ml/train.py resnets/resnet-6
```

where I omit the file extension.

This also uses WandB, so set up an account there.


## Friend Ntuples

To get the output of the model on our samples, we create friend ntuples, which are new ntuples that have only one branch: the output of the model which classifies the probability of an event being signal (LQ). This should also be done on the local machine. If you did the training on lxplus, then do this step also on lxplus, to keep it consistent.

We first need to copy all of the small ntuples to our local machine as well, and just as with the numpy files, place them in the same directory, namely `friend_ntuples/output/small`. All of the paths at this point should be relative, meaning that this is all that should have to be done. Now we just run

```
python3 friend_ntuples/produce_friend.py
```

The outputted files will be placed in `friend_ntuples/output/friend/<model-name>`.



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
