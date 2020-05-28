# cs461-cs5

## AI to predict Generative Design

### This github repo is for the Oregon State University Capstone Project with Autodesk


The repo is pretty straight forward where different directories represent different parts of our project.

- AWS is related to the server we are using to host the machine learning model

- Machine Learning relates to all the code that is being used to generate the model

- Fusion API is the code being used through Fusion360's API to communicate with our server on AWS

## Setup
Note that you cannot run the machine learning model because it is generated on data under an NDA but you can most certainly look at it if you want.

Setup is under assumption that you have Windows 10 as your default OS and you have Fusion360 installed with an account setup and you have Visual Studio Code installed.

1. clone or download the repo and open the directory (this can be done anywhere that is convenient for you)

2. Open up Fusion360

3. Within Fusion entered the environment and naviagate to the default window (usually the one you start up in)

4. Click the TOOLS button and then ckick the ADD-INS icon. Located on the top navigation bar.

5. Create a ADD-IN by clicking the create button.

6. Upon entering the Visual Studio Code when editing the ADD-IN copy the file UI_full_version.py into this new file. File is located in cs461-cs5/API/UI_full_version.py

7. Click the run button.

8. Modify the desired inputs to your liking (You can also activate this ADD-IN from the top navigation bar as well now)

9. Click finished and get your estimated time frame back!
