# test_voice.py
import pyttsx3

engine = pyttsx3.init()
engine.say("Hello testing voice")
engine.runAndWait()