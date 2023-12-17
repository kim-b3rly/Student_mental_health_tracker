import pandas as pd
from datetime import datetime
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import sqlite3

db_path = 'student_MH_data.db'

def create_table(answer_list1, answer_list2):
    # create a connection object with the database and SQLite
    conn = sqlite3.connect(db_path)

    # create a cursor to manipulate the tables and queries

    cursor = conn.cursor()

    # create a table to store user entries

    user_table = """ CREATE TABLE IF NOT EXISTS user_log (
    id INTEGER PRIMARY KEY, 
    log_date DATE NOT NULL,
    mood_log INTEGER,
    sleep INTEGER,
    uni_work INTEGER,
    other_work INTEGER,
    coping INTEGER,
    confidence INTEGER,
    struggles VARCHAR,
    contact_hrs VARCHAR,
    journal_entry TEXT,
    entry_score INTEGER
     );"""

    cursor.execute(user_table)

    # INSERT user inputs into the table

    # get the current date
    now = datetime.now()

    # Format the date as a string in the format "YYYY-MM-DD"
    date_string = now.strftime('%Y-%m-%d')

    cursor.execute("INSERT INTO user_log (log_date, uni_work, other_work, coping, confidence) VALUES (?, ?, ?, ?, ?)",
                   (date_string, answer_list2[0], answer_list2[1], answer_list2[2], answer_list2[3]))


# write a function to read the data from the table and create a dataframe

def read_table():
    with sqlite3.connect(db_path) as conn_obj:
        df = pd.read_sql_query('SELECT * FROM user_log ORDER BY log_date DESC', conn_obj)
    return df

# write a function for visualisations


def visualise_data():

    with sqlite3.connect(db_path) as conn_obj:
        df = read_table()

        # create a line plot for mood, sleep, coping
        # create a dataframe from the table
        df = pd.DataFrame(df)

        fig1 = px.line(df, x='log_date', y=['sleep', 'coping', 'uni_work', 'other_work'], line_shape='spline')

        fig1.update_layout(xaxis_tickformat='%Y-%m-%d')

        st.plotly_chart(fig1)

        # create a line plot for productivity



