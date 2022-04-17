
# Deep reinforcement learning for protein complex modeling


# Dependencies

* PyRosetta==4
* numpy>=1.20.2
* pandas>=1.2.5
* tqdm==4.61.1
* scipy>=1.6.2
* seaborn>=0.11.2
* setuptools>=44.0.0
* imageio>=2.10.1
* matplotlib>=3.4.2


# Installation

The package is tested using Python 3.6 and 3.7. To install the software, please follow the instructions below:

* Install the above dependencies
* Download and install PyRosetta (http://www.pyrosetta.org/dow)
* Install the package following the instruction below:


```
git clone git@github.com:jianlin-cheng/DeepRLP.git

(If fail, try username) git clone https://github.com/jianlin-cheng/DeepRLP.git

cd DeepRLP
pip install -r requirements.txt


Alternatively, environment.yml files are provided to install the required packages using pip or conda.
``` 


# Basic Usage

1. Reconstructing the dimer structure using true structure

```

python ./scripts/true_structure/dqn_docking_reward9.py <initial_pdb> <native_strcuture>

```

2. Reconstructing the dimer structure using true/predicted contacts

```

python ./scripts/predicted_contacts/dqn_docking.py <path_to_the_DeepRLP_tool> <initial_start> <res_file> <out_dir> <target_name>

```



A video demonstrating how DRLComplex reconstructs the dimer structure using true interchain contacts:


https://user-images.githubusercontent.com/45056120/163706448-ad3cb499-d9ec-423f-a36a-6d5932946f42.mp4



