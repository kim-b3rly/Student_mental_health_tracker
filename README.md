# Student_mental_health_tracker

### Introduction
This was my Master's thesis project, including research for the on-going mental health crisis among the NHS and universities where demand for therapies overwhelms exisitng capacity.
The proposed solution was an application that intergrates the self-managment main stressors and mental health triggers for university students into the framework of a mental well-being trackers.

### Artefact and natural language processing 
The module design of the artefact can be seen in the chart below. The cluster analysis was carried out on a student survey dataset to identify insights from clustering that would reveal themes of stress factors students expereince e.g exams, deadlines,financies, relationships e.t.c. Next, two sentiment classifiers (Support Vector Machine and LSTM) were trained for comparison and the best performing one was a bi-LSTM neural network. This was deployed as a journal that reports and records the sentiment in a separate database. 

![image](https://github.com/kim-b3rly/Student_mental_health_tracker/assets/99509204/95219019-e89a-4e46-9431-7093f427743f)

#### The database

A small databse was created using python SQL for the demo of the appliation to record and create a log of all entries made per day by the user. 

#### The GUI
Artefact was deployed using streamlit. Some features have not been full coded and linked to the database in the back-end, any contributions and suggestions are welcomed.
