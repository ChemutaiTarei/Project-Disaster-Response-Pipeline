# Disaster Response Pipeline Project

## Summary
The project involved analysing disaster data from Appen (formally Figure 8) to and building a model for an API that classifies disaster messages. A machine learning pipeline was created to categorize these disaster events. A web app was also created where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python Project_Disaster_Response_Pipeline/process_data.py Project_Disaster_Response_Pipeline/disaster_messages.csv Project_Disaster_Response_Pipeline/disaster_categories.csv Project_Disaster_Response_Pipeline/DisasterReponse.db`
    - To run ML pipeline that trains classifier and saves
        `python Project_Disaster_Response_Pipeline/train_classifier.py Project_Disaster_Response_Pipeline /DisasterReponse.db Project_Disaster_Response_Pipeline/model.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
