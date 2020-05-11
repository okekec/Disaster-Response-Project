# Disaster-Response-Project
## Getting Started
### Installation

- Python 3.6
- HTML
- Flask
- Plotly
- Python Packages:
  - Machine Learning libraries such as NumPy, SciPy, Pandas, Sciki-Learn
  - XGBoost
  - Python oversampling package such as imblearn
  - Regular expression operation python package such as re
  - Json
  - Natural Language Process libraries such as NLTK
  - SQLlite Database libraries such as SQLalchemy
  
## Project Overview  

The objective of this project is to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. The data set contains real messages that were sent during disaster events. A machine learning pipeline is created to categorize these events so that an emergency worker from a disaster relief agency can input a new message and get classification results in several categories.

### Project Details

#### Project Components
1. ETL Pipeline: In a Python script, *process_data.py*, write a data cleaning pipeline that:
      - Loads the messages and categories datasets
      - Merges the two datasets
      - Cleans the data
      - Stores it in a SQLite database

2. ML Pipeline: In a Python script, *train_classifier.py*, write a machine learning pipeline that:
    - Loads data from the SQLite database
    - Splits the dataset into training and test sets
    - Builds a text processing and machine learning pipeline
    - Trains and tunes a model using GridSearchCV
    - Outputs results on the test set
    - Exports the final model as a pickle file
  
3.  Flask Web App: In a Python script, *run.py*, write a Flask file that runs the disaster response app. In this file you will do the following:
    - Modify file paths for the database and model as needed
    - Add data visualizations using Plotly in the web app. 
  
#### File Descriptions
- ETL Pipeline Preparation.ipynb: 

  The first part of your data pipeline is the Extract, Transform, and Load process. In this jupyter notebook, read the dataset, clean the data, and then store it in a SQLite database. Use pandas for the data cleaning procedure. To load the data into an SQLite database, use the pandas dataframe `.to_sql()` method along with an SQLAlchemy engine.
  
- ML Pipeline Preparation.ipynb:

  This is the machine learning portion where the data is splited into a training set and a test set. A machine learning pipeline is created using NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). Finally, export the model to a pickle file. 

- process_data.py:

  This python file is created for your final cleaning code from the final ETL script in the ETL Pipeline Preparation notebook.
  
- train_classifier.py:

  This python file is created to include your final machine learning code after completing the ML Pipeline Preparation notebook.
  
#### Folder Descriptions
  - Data: 
  
    This folder contains messages and categories datasets in csv format. it also contains the SQLite database created from the ETL Pipeline Preparation notebook.
 
 - ML Model:
 
    This folder contains all the ETL and Machine learning procedures.
 
 - app:
 
    This folder contains all the files necessary to run and render the web app.
  
## Instructions for executing the program

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Running the Web App from the Project Workspace IDE
  When working in the Project Workspace IDE, here is how to see the Flask app.

  Open a new terminal window. You should already be in the workspace folder, but if not, then use terminal commands to navigate inside the folder with the run.py file.

  Type in the command line:

  `python run.py`

  Your web app should now be running if there were no errors. Now, open another Terminal Window.

  Type

  `env|grep WORK`

  After  running the code above, you will get a SPACE DOMAIN and ID. In a new web browser window, type in the following:

  `https://SPACEID-3001.SPACEDOMAIN`

  You should be able to see the web app. The number 3001 represents the port where your web app will show up. Make sure that the 3001 is part of the web address you type in.
  
## Images of the web app

Image 1: Intro page 

![Intro](https://github.com/okekec/Disaster-Response-Project/blob/master/app/Web%20app%20images/Intro%20page.JPG)

Image 2: Body page 

![Body](https://github.com/okekec/Disaster-Response-Project/blob/master/app/Web%20app%20images/Body%20page.JPG)

Image 3: Input

![Input](https://github.com/okekec/Disaster-Response-Project/blob/master/app/Web%20app%20images/input.JPG)

Image 4: App Result

![result](https://github.com/okekec/Disaster-Response-Project/blob/master/app/Web%20app%20images/App%20result.JPG)

## Image of the model result

![model](https://github.com/okekec/Disaster-Response-Project/blob/master/ML%20Model/model%20results.JPG)

## Licensing, Authors, Acknowledgements

This app was completed by me as part of the Udacity Data Scientist Nanodegree. Code templates and training were provided by Udacity. The data was originally sourced by Udacity from Figure Eight.
