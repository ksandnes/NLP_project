# NLP_project

Aim: to build an isolated-word predictor for a small vocabulary and test its success rate relative to matching adult vs child-voiced single-word recordings. 

Instructions to test the project: 

NOTE - due to data privacy issues, spoken word databank not included in this repository. Instructions are historical.

Download and unzip the project folder to an appropriate python3 run location on your device.

Edit the global variables ‘training_directory’ and ‘test_directory’ to reflect project folder location.

Dependencies: numpy, scipy, python_speech_features, sklearn (for specifics re those requirements, see code)

Run the program from the ‘book_project.py’ location in your terminal via the command options:

a.	python book_project.py knn         (to run in DTW-KNN prediction mode)

b.	python book_project.py dtw        (to run in DTW-only mode)

Your output should be a progressively printed list of word predictions for 4 age-labeled test subjects along with the correct word into the terminal output, ending with a percentage success rate for each subject. To view the progress of the KNN trainer (which does take a few minutes), uncomment line 70.
