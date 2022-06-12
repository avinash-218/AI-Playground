'''
Disclaimer:

DeepSphere.AI developed these materials based on its team’s expertise and technical infrastructure, and we are sharing these materials strictly for learning and research.
These learning resources may not work on other learning infrastructures and DeepSphere.AI advises the learners to use these materials at their own risk. As needed, we will
be changing these materials without any notification and we have full ownership and accountability to make any change to these materials.

Author :                          Chief Architect :       Reviewer :
____________________________________________________________________________
Avinash R & Jothi Periasamy       Jothi Periasamy         Jothi Periasamy
'''

import streamlit as st
import random
from gtts import gTTS
import os
import playsound
from PIL import Image
import speech_recognition as sr
import tensorflow as tf
from tensorflow import keras
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json
import warnings
warnings.filterwarnings('ignore')

def local_css(vAR_file_name):
    #function to apply style formatting from styles.css file in streamlit
    #filename - css file contains webpage formatting options
    with open(vAR_file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("./DSAI_Vbot_Style.css") #function call to load .css file and apply in streamlit webpage

vAR_col1, vAR_col2, vAR_col3 = st.columns([1,1,1])
vAR_col2.image('./DSAI_Vbot_DeepSphere_Logo.jpg', use_column_width=True)

st.title("Voice Based Chatbot") #set title for webpage

vAR_model = keras.models.load_model('../Utility/DSAI_Vbot_ANN_Trained_Model.h5')# load model

with open('../Utility/DSAI_Vbot_Tokenizer.pickle', 'rb') as f:
        vAR_tokenizer = pickle.load(f)

with open('../Utility/DSAI_Vbot_Label_Encoder.pickle', 'rb') as f:
    vAR_lbl_encoder = pickle.load(f)

with open('../Utility/DSAI_Vbot_Intents.json') as file:
    vAR_data = json.load(file)

vAR_max_len = 20

def chat():
    # method to chat with robot
    vAR_r1 = random.randint(1,10000000)
    vAR_r2 = random.randint(1,10000000)
    vAR_file = str(vAR_r2)+"randomtext"+str(vAR_r1) +".mp3" #generate random file name

    #Recording voice input using microphone 
    vAR_fst="Hello"
    vAR_tts = gTTS(vAR_fst,lang="en",tld="com") #google text to speech API to convert the message to audio
    vAR_tts.save(vAR_file) #save the audio with the random filename generated

    vAR_r = sr.Recognizer() #recognize user input
    st.write(f'<p style="font-family: sans-serif;font-size: 15px;text-transform: capitalize;background-color: #190380;padding: 18px;border-radius: 15px">{vAR_fst}</p>', unsafe_allow_html=True)
    playsound.playsound(vAR_file,True) #play the bot reply
    os.remove(vAR_file) #remove the file generated

    vAR_tag = ''
    while(vAR_tag!='goodbye'):
        with sr.Microphone(device_index=1) as source: #microphone as input device
            st.write("Listening...")
            vAR_audio= vAR_r.adjust_for_ambient_noise(source)
            vAR_audio = vAR_r.listen(source)
        try:
            vAR_user_response = format(vAR_r.recognize_google(vAR_audio))
            st.write(f'<p style="font-family: sans-serif;color: white;font-size: 15px;text-align:right;text-transform: capitalize;background-color: #190380;padding: 18px;border-radius: 15px">{vAR_user_response}</p>', unsafe_allow_html=True)

            t = vAR_tokenizer.texts_to_sequences([vAR_user_response])
            p = pad_sequences(t, truncating='post', maxlen=vAR_max_len)
            vAR_result = vAR_model.predict(p)
            vAR_pred_class = np.argmax(vAR_result)
            vAR_tag = vAR_lbl_encoder.inverse_transform([vAR_pred_class])

            for i in vAR_data['intents']:
                if i['tag'] == vAR_tag:
                    vAR_resp = np.random.choice(i['responses'])

        except sr.UnknownValueError:
            vAR_resp = "Oops! Didn't catch that"

        st.write("Bot:")
        st.write(f'<p style="font-family: sans-serif;font-size: 15px;text-transform: capitalize;background-color: #190380;padding: 18px;border-radius: 15px">{vAR_resp}</p>', unsafe_allow_html=True)
        vAR_tts = gTTS(vAR_resp,tld="com")
        vAR_tts.save(vAR_file)
        playsound.playsound(vAR_file,True)
        os.remove(vAR_file)

if st.button("Get your Assistant"): #event listener for 'Get your Assistant' button
    chat()

image2 = Image.open('../User_Interface/DSAI_Vbot_Robot.png')
st.image(image2)

'''
Copyright Notice:

Local and international copyright laws protect this material. Repurposing or reproducing this material without written approval from DeepSphere.AI violates the law.

(c) DeepSphere.AI
'''