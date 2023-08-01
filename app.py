import webbrowser
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
from streamlit_card import card
from streamlit_extras.let_it_rain import rain


model  = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.header("Emotion Identification and Music Recommendation on Spotify")
flag = True

if "run" not in st.session_state:
	st.session_state["run"] = "true"

try:
	emotion = np.load("emotion.npy")[0]
except:
	emotion=""

if not(emotion):
	st.session_state["run"] = "true"
else:
	st.session_state["run"] = "false"



class EmotionProcessor:
	def recv(self, frame):
		frm = frame.to_ndarray(format="bgr24")

		frm = cv2.flip(frm, 1)

		res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

		lst = []

		if res.face_landmarks:
			for i in res.face_landmarks.landmark:
				lst.append(i.x - res.face_landmarks.landmark[1].x)
				lst.append(i.y - res.face_landmarks.landmark[1].y)

			if res.left_hand_landmarks:
				for i in res.left_hand_landmarks.landmark:
					lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
					lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
			else:
				for i in range(42):
					lst.append(0.0)

			if res.right_hand_landmarks:
				for i in res.right_hand_landmarks.landmark:
					lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
					lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
			else:
				for i in range(42):
					lst.append(0.0)

			lst = np.array(lst).reshape(1,-1)

			pred = label[np.argmax(model.predict(lst))]

			print(pred)
			cv2.putText(frm, pred, (40,60),cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,0),2)

			np.save("emotion.npy", np.array([pred]))

			
		drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
		drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
		drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

		return av.VideoFrame.from_ndarray(frm, format="bgr24")

lang = st.radio("Language", ["English", "Hindi", "Punjabi", "Kannada"])
singer = st.text_input("Singer Name")




if lang and singer and st.session_state["run"] != "false":
	flag = False
	webrtc_streamer(key="key", desired_playing_state=True,
				video_processor_factory=EmotionProcessor)

btn = st.button("Recommend me songs")

if btn:
	if not(emotion):
		st.warning("Hola, Give Me some Time to capture your Emotion ;-)")
		st.session_state["run"] = "true"
	else:
		flag = False
		webbrowser.open (f"https://open.spotify.com/search/{singer}%20{emotion}%20song%20{lang}")
		np.save("emotion.npy", np.array([""]))
		st.session_state["run"] = "false"
if(flag):
	rain(
			emoji="😁",
			font_size=40,  
			falling_speed=9,  
			animation_length=200, 
		)
	rain(
			emoji="😡",
			font_size=40,
			falling_speed=3,  
			animation_length=200, 
		)
card(
    title="Spotify",
    text="Open Spotify Directly",
    image= "https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Spotify_icon.svg/1982px-Spotify_icon.svg.png",
    url="https://open.spotify.com/",
    styles={
        "card": {
            "width": "200px",
            "height": "200px",
            "border-radius": "100px",
            "box-shadow": "0 0 50px rgba(30,215,96,5)",
	    	"background-color": "#f2f2f2",
    		"padding": "20px"
           
        },
        "text": {
            "font-family": "sans-serif",
        }
	}
)
