# importing libraries
import pandas as pd
import datetime as dt
import streamlit as st
import plotly.graph_objects as go
from streamlit_image_select import image_select
from PIL import Image
import joblib
import numpy as np
import nltk
import keras
import tensorflow as tf
import re
import pickle
import tracker_db

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from tracker_db import *

# setting constants

stopwords_list = set(stopwords.words('english'))

# Load the embeddings dictionary from the saved file

# with open('embeddings_dict.pkl', 'rb') as file:
#     loaded_embeddings_dict = pickle.load(file)

# loading the saved text classification model

sentiment_model = keras.models.load_model('BiLSTM.keras')

# this function will take the journal entry to predict the mood of the user

# the user's journal text will be analysed and the score will be kept


def journal_prediction(clean_journal_text):

    sentiment = sentiment_model.predict(clean_journal_text)

    if sentiment < 0.5:
        sentiment = 0
        # log mood as not stressed
    else:
        sentiment = 1
        # log mood as stressed

    return sentiment


# create an object for the lemmatization method

wnl = WordNetLemmatizer()
word_tokenizer = Tokenizer()


def preprocess(journal_text):
    # special character, number  removal

    journal_text = re.sub(r"[^a-zA-Z]", " ", journal_text)

    # convert all text to lower case

    journal_text = journal_text.lower()

    # create an object to compile all the stopwords for matching them to the input text

    pattern = re.compile(r'\b(' + r'|'.join(stopwords_list) + r')\b\s*')

    # replace stopwords with an empty space

    text = pattern.sub('', journal_text)

    # tokenization, ie split it into a list of unique words

    tokens = word_tokenize(text)

    # lemmatize each word from the tokens list and join them back into a sentence

    lemmatized_text = ' '.join([wnl.lemmatize(t) for t in tokens])

    # convert the text to sequences and add padding

    # Convert the lemmatized text to a list with one element
    text_list = [lemmatized_text]

    word_tokenizer.fit_on_texts(text_list)

    text_sequence = word_tokenizer.texts_to_sequences(text_list)
    # st.write(f"pre padding sequence {text_sequence}")

    text_length = len(text_sequence)
    text_sequence = tf.keras.utils.pad_sequences(text_sequence, maxlen=300,
                                                 dtype='int32',
                                                 padding='post',
                                                 truncating='post',
                                                 value=0.0)
    # st.write(f"post padding sequence {text_sequence}")

    return text_sequence


user_answers = []


def main():
    menu = ["Home", "Well-being", "Academic tracker", "Journal"]

    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.header("Home")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Daily Goals")
            # records how many have been completed in one day
            # set goals in this page

        with col2:
            st.subheader("Mood Log")
            # records the streak of mood logged per day or a plot

        with col3:
            st.subheader("Stress Log")
            # plots stress levels

    elif choice == "Well-being":

        st.header("Track your well-being ")

        st.write("Your mental health is affected by a variety of factors. It is easier to pin point and manage these if you regularly log your triggers, mood and coping strategies")

        st.subheader("Mood")

        # img = image_select(
        #     label="How are you feeling today?",
        #     images=[
        #         Image.open(r"C:\Users\kimb3\OneDrive - Solent University\Desktop\disso\GUI images\mood resized\slight_smile.png"),
        #         Image.open(r"C:\Users\kimb3\OneDrive - Solent University\Desktop\disso\GUI images\mood resized\worried.png"),
        #
        #     ],
        #     captions=["okay","worried"]
        # )

        if st.button(":grinning:", use_container_width=True):
            user_answers.append(7)
        if st.button(":smile:"):
            st.text("Good job logging your mood!")
            user_answers.append(6)
        if st.button(":relieved:"):
            user_answers.append(5)
        if st.button(":neutral_face:"):
            user_answers.append(4)
        if st.button(":tired_face:"):
            user_answers.append(3)
        if st.button(":cry:"):
            user_answers.append(2)
        if st.button(":worried:"):
            user_answers.append(1)
        if st.button(":sweat:"):
            user_answers.append(0)

        st.subheader("Hours slept")
        if st.button("at most 5"):
            st.text(" ")
        if st.button(" 5-9"):
            st.text(" ")
        st.button("at least 10")

    elif choice == "Academic tracker":

        st.header("Stay on top of your books ")

        def ask_questions():
            st.subheader("productivity and workload")
            # Define the questions
            work_questions = [
                "How productive were you with school work today? (0 = next question, 10 = crushing it)",
                "How productive were you with non-school related tasks today?(0 = not at all, 10 = ultimate adulting)",
                "Are you able to cope with your current workload?(0= I'm very behind, 10 = doing great)",
                "Are you confident about any upcoming deadlines or exams? (0 = meh, 10 = definitely)"

            ]

            work_answers = []

            # Ask the questions and get the answers
            for question in work_questions:
                answer = st.slider(question, 0, 10)
                work_answers.append(answer)

            return work_answers

        a_answers = ask_questions()

        # insert the answers into the database

        save_to_db = create_table(user_answers, a_answers)


        col1_t3, col2_t3, col3_t3 = st.columns(3)
        with col1_t3:

            st.subheader("Struggles")
            if st.button("Course difficulty"):
                st.text("Options")
            if st.button("Health flare up"):
                st.text("Here are some suggestions...")
            if st.button("I need to catch up"):
                st.text("Here are some planning tools!")
            if st.button("Procrastination"):
                st.text("Did you know procrastination is not just due to laziness but...")

        with col2_t3:
            st.subheader("Contact hours")
            if st.button("Skipped"):
                st.text(" ")
            if st.button("Attended some"):
                st.text(" ")
            st.button("No classes today")
            st.button(" ")

        with col3_t3:
            st.subheader("Focus sessions")
            if st.button("Start session"):
                st.text("Session starting...")

    elif choice == "Journal":

        st.header("My journal")

        journal_text = st.text_area("whether you need to vent or reflect, don't bottle it up!", " ")

        if st.button("Save"):

            clean_journal_text = preprocess(journal_text)

            journal_sentiment = journal_prediction(clean_journal_text)

            if journal_sentiment == 0:
                st.write(" ")
            else:
                st.write(" Would you like to log your mood?")
                st.button("Log mood")

            visualise_data()


if __name__ == '__main__':
    main()
