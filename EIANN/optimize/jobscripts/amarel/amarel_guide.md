# Guide to Optimization on Amarel

Clone EIANN and nested into amarel login node. Switch to pre_release branch for nested. 
```bash
$ git clone https://github.com/Milstein-Lab/EIANN.git
$ git clone https://github.com/neurosutras/nested.git
$ cd nested
nested$ git checkout pre_release
```

Make conda environment for EIANN
```bash
nested$ cd ../EIANN
EIANN$ conda create --name eiann python=3.9
EIANN$ conda activate eiann
EIANN$ pip install -r requirements.txt
EIANN$ conda install anaconda::mpi4py
```

Load required modules
```bash
EIANN$ cd ..
$ module use /projects/community/modulefiles
$ module load openmpi/4.1.6
```

Now to make jobscript, model it on EIANN/EIANN/optimize/jobscripts/optimize_EIANN_amarel.sh 

To navigate between Home and Scratch directories:
```bash
$ cd /home/$USER
```
```bash
$ cd /scratch/$USER
```