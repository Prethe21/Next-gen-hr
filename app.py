from flask import Flask, render_template, request
import PyPDF2
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import random
import cv2
import os
from datetime import datetime
import subprocess
import sqlite3
import sys
import os
import numpy as np
from flask import Flask, request, render_template, flash, redirect
import math
from collections import Counter
import re
import random
import csv
import ast
from difflib import SequenceMatcher
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from werkzeug.utils import secure_filename
import shutil
from tensorflow.keras.models import model_from_json
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from matplotlib import pyplot as plt
import io
import base64
import urllib
from skimage.metrics import structural_similarity as compare_ssim
from moviepy import VideoFileClip


app = Flask(__name__)


conn = sqlite3.connect('user_database.db')

cursor = conn.cursor()


cursor.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, mail TEXT NOT NULL UNIQUE, password TEXT NOT NULL)')
cursor.execute('CREATE TABLE IF NOT EXISTS admins (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, mail TEXT NOT NULL UNIQUE, password TEXT NOT NULL)')
cursor.execute('CREATE TABLE IF NOT EXISTS user_dat_em (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, mail TEXT NOT NULL, emotion_data TEXT NOT NULL, mark INTEGER NOT NULL)')
cursor.execute('CREATE TABLE IF NOT EXISTS feedbacks (id INTEGER PRIMARY KEY AUTOINCREMENT, mail TEXT NOT NULL, content TEXT NOT NULL)')
cursor.execute('CREATE TABLE IF NOT EXISTS company (id INTEGER PRIMARY KEY AUTOINCREMENT, company TEXT NOT NULL Unique, role TEXT NOT NULL, place TEXT NOT NULL, descriptions TEXT NOT NULL)')

conn.commit()
conn.close()
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')   #load face detection cascade file

@app.route('/admin', methods=['GET','POST'])
def admin():
    return render_template('admin.html')

@app.route('/register_admin', methods=['GET','POST'])
def register_admin():
    if request.method == 'POST':
        username = request.form['username']
        mail = request.form['mail']
        password = request.form['pass']
        conn = sqlite3.connect('user_database.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO admins (username, mail, password) VALUES (?, ?, ?)', (username, mail, password))
        conn.commit()
        return render_template('admin.html')
    return render_template('admin.html')

@app.route('/login_admin', methods=['GET','POST'])
def login_admin():
    global userid
    if request.method == 'POST':
        userid = request.form['userid']
        password = request.form['pass']
        conn = sqlite3.connect('user_database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM admins WHERE mail=? AND password=?', (userid, password))
        user = cursor.fetchone()
        if user:
            return render_template('admin_home.html')
        else:
            return render_template('admin.html')
    return render_template('admin.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        mail = request.form['mail']
        password = request.form['pass']
        conn = sqlite3.connect('user_database.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (username, mail, password) VALUES (?, ?, ?)', (username, mail, password))
        conn.commit()
        return render_template('login.html')
    return render_template('login.html')

@app.route('/login', methods=['GET','POST'])
def login():
    global userid
    if request.method == 'POST':
        userid = request.form['userid']
        password = request.form['pass']
        conn = sqlite3.connect('user_database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE mail=? AND password=?', (userid, password))
        user = cursor.fetchone()
        if user:
            
            return render_template('index.html')
        else:
            return render_template('login.html')
    return render_template('login.html')

vacancies = []


@app.route('/post_vacancy', methods=['POST','GET'])
def post_vacancy():
    if request.method == 'POST':
        role = request.form['role']
        company = request.form['company']
        Place = request.form['Place']
        description = request.form['description']
        conn = sqlite3.connect('user_database.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO company (company, role, place, descriptions) VALUES (?, ?, ?, ?)', (company, role, Place, description))
        conn.commit()
        return render_template('admin_home.html')
    return render_template('admin_home.html')

    

##@app.route('/user_home')
##def user_home():
##    conn = sqlite3.connect('user_database.db')
##    cursor = conn.cursor()
##    cursor.execute('SELECT company, role, place, descriptions FROM company')
##    vacancies = cursor.fetchall()
##    conn.close()
##    
##    return render_template('user_home.html', vacancies=vacancies)


@app.route('/aptitude_test')
def aptitude_test():
    global company, role
    print('work')
    company = request.args.get('company')
    print(company)
    role = request.args.get('role')
    print(role)
    return render_template('aptitude_test.html', company=company, role=role)


@app.route('/add', methods=['POST'])
def add():
    question = request.form['ques']
    lang = request.form['language']
    answer = request.form['ans']
    ans = re.sub(r'[^\w\s]','', answer) 
        # Tokenize the string
    tokens = word_tokenize(ans)
    # Remove stopwords (common words that don't carry much meaning)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        
    print(filtered_tokens)
    print(type(filtered_tokens))
        
    orig_ques=question
    keywords=filtered_tokens
    with open('keywords.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([orig_ques, lang, keywords])
    return 'add sucessfully'
    

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    num_pages = len(pdf_reader.pages)
    text = ""
    for page_num in range(num_pages):
        text += pdf_reader.pages[page_num].extract_text()
    return text

def extract_skills(resume_text, num_skills=3):
    words = word_tokenize(resume_text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

    # Add your own list of skills or use an existing one
    skills_list = ['python', 'java', 'machine learning', 'communication', 'teamwork', 'data analysis', 'project management']

    extracted_skills = [word for word in filtered_words if word.lower() in skills_list][:num_skills]
    return extracted_skills

def get_questions_by_skill(questions_df, skill):
    # Filter questions based on the skill
    print(questions_df.columns.tolist())

    # Try to filter questions based on the skill
    try:
        filtered_questions = questions_df[questions_df['Skill'].str.lower() == skill.lower()][['Question', 'Answer']].to_dict('records')
        return filtered_questions
    except KeyError as e:
        # Handle the KeyError and print an error message
        print(f"KeyError: {e}")
        return []

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global skills
    if 'resume' in request.files:
        resume = request.files['resume']
        resume_text = extract_text_from_pdf(resume)
        skills = extract_skills(resume_text, num_skills=1)
        print(skills)# Extract 1 skill
        if skills:
            conn = sqlite3.connect('user_database.db')
            cursor = conn.cursor()
            
            # Use LIKE to check if the role contains the skill
            cursor.execute('SELECT company, role, place, descriptions FROM company WHERE LOWER(role) LIKE ?', (f"%{skills[0].lower()}%",))
            matching_vacancies = cursor.fetchall()
            
            conn.close()

            if not matching_vacancies:
                return "No matching company found for your skills!", 403  # Forbidden
            
            return render_template('user_home.html', vacancies=matching_vacancies)
        else:
            return render_template('index.html')

        
    return render_template('result.html', error='No file uploaded')

def generate_answers(questions):
    answers = [f"Answer for {question}" for question in questions]

    return answers

def save_questions_and_answers(output_csv, questions):
    answers = generate_answers(questions)  # Implement generate_answers based on your logic
    df = pd.DataFrame({'Question': questions, 'Answer': answers})
    df.to_csv(output_csv, index=False)


def get_question_numbers(filename):
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            question_numbers = {}
            for row in reader:
                question_numbers[row[0]] = row[1]
        return question_numbers

def get_questions(filename, num_questions, desired_skills):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        questions = [(row[0], row[1]) for row in rows]

    if isinstance(desired_skills, str):
        desired_skills = [desired_skills]
    filtered_questions = [question[0] for question in questions if question[1] in desired_skills]
    random.shuffle(filtered_questions)
    return filtered_questions[:num_questions]


def create(questions,question_numbers):
        global original_question_order
        random.shuffle(questions)
        original_question_order = {question: question_numbers[question] for question in questions}
        print(original_question_order)

def save_answers_to_csv(original_order, answers, output_file='answers.csv'):
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Question', 'OriginalOrder', 'Answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for question, original_order in original_order.items():
            answer = answers[question]  # Assuming answers is a dictionary with questions as keys
            writer.writerow({'Question': question, 'OriginalOrder': original_order, 'Answer': answer})
            print('saved')

import csv

def get_answers_from_csv(file_path):
    answers = {}

    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        
        # Skip the header row
        next(reader, None)

        for row in reader:
            question = row[0]  # Assuming the question is in the first column
            answer = row[2]     # Assuming the answer is in the third column
            answers[question] = answer

    return answers


@app.route('/question', methods=['GET','POST'])
def question():
    global questions
    global question_numbers
    num_questions = 10
    skill = skills
    print(skill)
    questions = get_questions('keywords.csv', num_questions, skill)
    question_numbers = get_question_numbers('keywords.csv')
    create(questions, question_numbers)
    answers = get_answers_from_csv('keywords.csv')  # Implement this function
    
    save_answers_to_csv(original_question_order, answers)

    return render_template('questions.html', skill=skill, questions=questions)


import speech_recognition as sr

from moviepy.video.io import ffmpeg_tools
#from moviepy.editor import VideoFileClip

def separate_and_transcribe_videos(video_folder, output_excel):
    recognizer = sr.Recognizer()
    results = []
    VideoFileClip.DEFAULT_EXECUTABLE = "ffmpeg"


    # Ensure the output folder exists
    output_folder = 'static/audio'
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through each video file in the specified folder
    for video_file in os.listdir(video_folder):
        if video_file.endswith(".webm"):  # Adjust the extension as needed
            video_path = os.path.join(video_folder, video_file)

            # Extract audio from the video
            audio_output_path = os.path.join(output_folder, f'{video_file.split(".")[0]}_audio.wav')
            ffmpeg_tools.ffmpeg_extract_audio(video_path, audio_output_path)

            # Transcribe the audio
            with sr.AudioFile(audio_output_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)

                # Append the result to the list
                results.append({'Video': video_file, 'Transcript': text})

    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    data = combinedvalue()    
    #print('Transcription completed. Results saved in', output_excel)
    return data

def combinedvalue():
    file1_path = "answers.csv"
    file2_path = "output_transcriptions.xlsx"
    df1 = pd.read_csv(file1_path, encoding='latin1')
    df2 = pd.read_excel(file2_path)
    result_df = pd.concat([df1, df2], axis=1)
    output_csv_path = "output_combined.csv"
    result_df.to_csv(output_csv_path, index=False)
    result_df = pd.read_csv("output_combined.csv")
    # Apply your text preprocessing to the 'Transcript' column
    result_df['Transcript'] = result_df['Transcript'].apply(lambda answer: re.sub(r'[^\w\s]', '', str(answer)))

    # Tokenize the string
    result_df['Transcript'] = result_df['Transcript'].apply(lambda ans: word_tokenize(ans))

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    result_df['Transcript'] = result_df['Transcript'].apply(lambda tokens: [word.lower() for word in tokens if word.lower() not in stop_words])

    # Convert the list of filtered tokens into a string of keywords
    result_df['Keywords'] = result_df['Transcript'].apply(lambda filtered_tokens: ' '.join(filtered_tokens))

    # Optionally, you can drop the intermediate 'Transcript' if you don't need it anymore
    result_df = result_df.drop(columns=['Transcript'])

    # Save the updated DataFrame to a new CSV file
    result_df.to_csv("output_with_keywords.csv", index=False)    

    special_chars = re.compile(r'[^a-zA-Z0-9 ]+')

    # Open the CSV file
    with open('output_with_keywords.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = [row[:6] for row in reader]  # Extract the first three columns of each row
    # Remove special characters from the text in the first three columns
    clean_data = [[special_chars.sub('', cell).lower() for cell in row] for row in data]

    # Compare the text in the three columns using the SequenceMatcher algorithm
    total_ratio = 0.0
    for row in clean_data:
        ratio = SequenceMatcher(None, row[2], row[4]).ratio()  # Compare column 1 with column 2
        total_ratio += ratio


    avg_ratio = total_ratio / (len(clean_data) * 3)  # Multiply by the number of comparisons (3 in this case)

    # Convert the ratio to a percentage
    percent_match = round(avg_ratio * 100, 2)
    print(percent_match)
    return percent_match

    


with open("model.json", "r") as json_file:   #Loading the saved model
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model_weights.h5")
loaded_model.make_predict_function()
label_to_text = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4: 'sad'}

def pred(img_path):  
    label_to_text = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4: 'sad'}  
    img=cv2.imread(img_path)									#read Image
    gray_fr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)				#covert image to grayscale
    faces_rects = facec.detectMultiScale(gray_fr, scaleFactor = 1.2, minNeighbors = 5)  #opencv's cascade classifier will be used for detecting the face
    if len(faces_rects)!=0:
        for (x, y, w, h) in faces_rects:
            fc = gray_fr[y:y+h, x:x+w]     #extracting only the face part
        roi = cv2.resize(fc, (48, 48))	#resizing it according to the image that are acceptable by our model
        img = image.img_to_array(roi)
        img = img/255
        img = np.expand_dims(img, axis=0)
        return label_to_text[np.argmax(loaded_model.predict(img))],img  
    else:
        return 0,0  


def removeout():
    shutil.rmtree('output/')
    
def vidframe(video_files):
        video_folder, video_file = video_files
        vidname = os.path.join(video_folder, video_file)
        if os.path.exists('output'):
            removeout()
        os.mkdir('output')
        cap = cv2.VideoCapture(vidname)
        frameRate=cap.get(5)
        count = 0
        while(cap.isOpened()):
            frameId = cap.get(1)
            ret, frame = cap.read()
            if (ret != True):
                break
            if (frameId % math.floor(frameRate) == 0):
                filename ="output/frame%d.jpg" % count;count+=1
                cv2.imwrite(filename, frame)
        cap.release()
        result=[]
        face=[]
        for filename in os.listdir("output"):
            a,b = pred("output/"+filename)
            result.append(a)
            face.append(b)
        removeout()
        result=[x for x in result if x!=0]
        face=[x for x in face if len(str(x))>1]
        return result, face



@app.route('/save_video', methods=['POST'])
def save_video():
    video_file = request.files.get('video')

    if video_file:
        # Save the video to the static folder
        video_file.save('static/video/' + video_file.filename)
        return 'Video saved successfully', 200
    else:
        return 'Failed to save video', 400

@app.route('/uploads', methods=['POST'])
def uploads():
    
    video_folder = 'static/video/'
    video_files = (video_folder, "recorded_video_3.webm") 
   
    result, face = vidframe(video_files)
    print(result)
    emotion_counts = Counter(result)
    most_common_emotion = emotion_counts.most_common(1)[0][0]
    print(most_common_emotion)
    fig = plt.figure()     #matplotlib plot
    ax = fig.add_axes([0,0,1,1])
    ax.axis('equal')
    emotion = ['angry','disgust','fear', 'happy', 'sad']
    counts = [result.count('angry'),result.count('disgust'),result.count('fear'),result.count('happy'),result.count('sad')]
    print(counts)
    ax.pie(counts, labels = emotion,autopct='%1.2f%%')   #adding pie chart
    img = io.BytesIO()
    plt.savefig(img, format='png')   #saving piechart
    img.seek(0)
    plot_data = base64.b64encode(img.read()).decode()
    output_excel_path = 'output_transcriptions.xlsx'
    value = separate_and_transcribe_videos(video_folder, output_excel_path)
    conn = sqlite3.connect('user_database.db')
    cursor = conn.cursor()
    query = "SELECT username FROM users WHERE mail = ?"
    cursor.execute(query, (userid,))
    result = cursor.fetchone()
    print(type(result))
    print(type(counts))
    print(type(userid))
    cursor.execute('INSERT INTO user_dat (username, mail, emotion_data, mark) VALUES (?, ?, ?, ?)', (result[0], userid, plot_data, value))
    conn.commit()
    return render_template('final.html')


@app.route('/final',methods=['GET','POST'])
def final():
    return render_template('final.html')

@app.route('/main',methods=['GET','POST'])
def main():
    return render_template('main.html')


@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback_content = request.form.get('feedback')

    if feedback_content:
        # Store the feedback in the database
        conn = sqlite3.connect('user_database.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO feedbacks (mail, content) VALUES (?,?)', (userid, feedback_content,))
        conn.commit()
        conn.close()
        return render_template('final.html')  # Redirect to a success page
    else:
        return "Error: Feedback content is empty."

@app.route("/submit_score", methods=["POST"])
def submit_score():
    data = request.json
    role = data["role"]
    company = data["company"]
    score = data["score"]

    # Store in database
##    conn = sqlite3.connect("aptitude_test.db")
##    cursor = conn.cursor()
##    cursor.execute("INSERT INTO scores (role, company, score) VALUES (?, ?, ?)", (role, company, score))
##    conn.commit()
##    conn.close()

    # Check score and send response
    if score >= 6:
        return jsonify({"message": "Congratulations! You passed. Redirecting to the coding test."})
    else:
        return jsonify({"message": "Your score is too low. Please try again."})

@app.route("/data", methods=['GET','POST'])
def data():
    return render_template('data.html')

# Route for coding test
@app.route("/coding_test")
def coding_test():
    global company, role
    print(company)
    print(role)
    #company = request.args.get('company', 'Unknown Company')  # Default to 'Unknown Company'
    #role = request.args.get('role', 'Unknown Role')  # Default to 'Unknown Role'
    print(f"Company: {company}, Role: {role}")  # Debugging log
    return render_template("coding_test.html", company=company, role=role)

@app.route('/user_answer')
def user_answer():
    # Fetch user data from the database
    conn = sqlite3.connect('user_database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM user_dat')
    user_data = cursor.fetchall()
    conn.close()

    return render_template('user_answer.html', user_data=user_data)

@app.route('/feedback_answer')
def feedback_answer():
    # Fetch user data from the database
    conn = sqlite3.connect('user_database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM feedbacks')
    user_data = cursor.fetchall()
    conn.close()

    return render_template('feedback_answer.html', user_data=user_data)


if __name__ == '__main__':
    app.run(debug=False, port=440)
