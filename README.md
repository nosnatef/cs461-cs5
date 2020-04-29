# cs461-cs5

## AI to predict Generative Design

### This github repo is for the Oregon State University Capstone Project with Autodesk


The repo is pretty straight forward where different directories represent different parts of our project.

- AWS is related to the server we are using to host the machine learning model

- Machine Learning relates to all the code that is being used to generate the model

- Fusion API is the code being used through Fusion360's API to communicate with our server on AWS

## Setup
0. Setup is under assumption that you have Linux as your OS. If not refer to documentation of similar linux commands to your OS

1. clone or download the repo and open the directory

If you want to setup to run the machine learning part either:

A)  pip install json, numpy, matplotlib

OR

B)  
1. install conda

2. enter the machine_learning/ folder
    
3. type: conda env create --file environment.yaml
    
4. type: conda activate cap_env  
        Note: This env name could possibly be different

To test the dataset (within machine_learning/ directory) type: python analyze_dataset.py


If you want to run the Fusion API:

1. Download the Autodesk Fusion 360 and create a account.

2. Click the TOOLS button and then ckick the ADD-INS icon.

3. Create a new py file and use Visual Studio Code onen it, then copy the code into the file.

4. Click the run button.

