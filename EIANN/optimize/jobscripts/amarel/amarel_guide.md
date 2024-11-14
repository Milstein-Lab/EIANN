# Guide to Optimization on Amarel

## Setup Git Repositories

Clone EIANN and nested into amarel login node. Switch to pre_release branch for nested. 
```bash
$ git clone https://github.com/Milstein-Lab/EIANN.git
$ git clone https://github.com/neurosutras/nested.git
$ cd nested
nested$ git checkout pre_release
```

## Dependencies

Setup conda environment and local installation for EIANN
```bash
nested$ cd ../EIANN
EIANN$ conda create --name eiann python=3.9
EIANN$ conda activate eiann
EIANN$ pip install -r requirements.txt
EIANN$ conda install anaconda::mpi4py
EIANN$ pip install --target ~/miniconda/envs/eiann/lib/python3.9/site-packages -e .
```

## Modules

Load required modules
```bash
EIANN$ cd ..
$ module use /projects/community/modulefiles
$ module load openmpi/4.1.6
```

## Editing .bashrc

This above procedure would require you to type that all out every time you ssh into Amarel. Instead, you can add instructions to the .bashrc file so that the conda environment is automatically activated and the required modules are loaded. First, open up the file:
```bash
vim ~/.bashrc
i
```

Then, add this to the end of the file:
```bash
# Load OpenMPI and activate conda environment when in EIANN directory
if [[ "$PWD" == *"/EIANN" ]]; then
    module use /projects/community/modulefiles
    module load openmpi/4.1.6
    if [[ "$CONDA_DEFAULT_ENV" != "eiann" ]]; then
        conda activate eiann
    fi
fi

# Add EIANN and nested directories to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/home/<user>
```
Where ```<user>``` is the Amarel username (found with ```echo $USER```).

Now, save and quit the vim session.
```
<esc>
:wq
```

Save the changes:
```bash
source ~/.bashrc
```

## Making a jobscript

Now to make jobscript, model it on EIANN/EIANN/optimize/jobscripts/optimize_EIANN_amarel.sh 

## Navigation

To navigate between Home and Scratch directories:
```bash
$ cd /home/$USER
```
```bash
$ cd /scratch/$USER
```