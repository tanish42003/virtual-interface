import cv2
import mediapipe as mp
import pyautogui
import speech_recognition as sr
import pyttsx3
import datetime
import wikipedia
import pywhatkit as kit
import webbrowser
import time
import os
import subprocess
from ecapture import ecapture as ec
import wolframalpha
import pyaudio
import requests
from pynput.mouse import Button, Controller
import numpy as np

def get_angle(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle


def get_distance(landmark_ist):
    if len(landmark_ist) < 2:
        return
    (x1, y1), (x2, y2) = landmark_ist[0], landmark_ist[1]
    L = np.hypot(x2 - x1, y2 - y1)
    return np.interp(L, [0, 1], [0, 1000])

mouse = Controller()

screen_width, screen_height = pyautogui.size()

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

recognizer = sr.Recognizer()

symbol_mapping = {
    "coma": ",",
    "full stop": ".",
    "question mark": "?",
    "exclamation mark": "!",
    "at the rate": "@",
    "hash symbol": "#",
    "percent symbol": "%",
    "space": " ",
    "new line": "\n",
}


def capture_audio():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio).lower()
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
        except sr.RequestError:
            print("Sorry, the speech service is unavailable.")
        return None

def open_application():
    speak("Which application would you like to open?")
    app_name = takeCommand().lower()
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    found = False

    for file in os.listdir(desktop_path):
        if app_name in file.lower():
            app_path = os.path.join(desktop_path, file)
            os.startfile(app_path)
            speak(f"Opening {app_name}")
            found = True
            break

    if not found:
        speak("Sorry, I couldn't find that application on the desktop.")


def type_text(text):
    if text:
        if text in symbol_mapping:
            pyautogui.write(symbol_mapping[text])
        else:
            pyautogui.write(text)



engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
        try:
            statement = r.recognize_google(audio, language='en-in')
            print(f"User said: {statement}\n")
        except Exception as e:
            speak("Please say that again")
            return "None"
        return statement.lower()

def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
        index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        return (index_finger_tip.x, index_finger_tip.y)
    return None

def move_mouse(index_finger_tip):
    if index_finger_tip is not None:
        x = int(index_finger_tip[0] * 1.4 * screen_width)
        y = int(index_finger_tip[1] * screen_height)
        pyautogui.moveTo(x, y)

def is_left_click(landmark_list, thumb_index_dist):
    return (
            get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
            thumb_index_dist > 50
    )

def is_right_click(landmark_list, thumb_index_dist):
    return (
            get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90 and
            thumb_index_dist > 50
    )

def is_double_click(landmark_list, thumb_index_dist):
    return (
            get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            thumb_index_dist > 50
    )

def is_voice_activation(landmark_list, thumb_index_dist):
    if len(landmark_list) < 13:  # Ensure the required landmarks exist
        return False
    return (
        get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        thumb_index_dist < 50
    )

def is_voice_keyboard(landmark_list):
    return (
        get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90 and  # Index finger extended
        get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and  # Middle finger extended
        get_angle(landmark_list[13], landmark_list[14], landmark_list[16]) > 90 and  # Ring finger extended
        get_angle(landmark_list[4], landmark_list[3], landmark_list[2]) > 50 and  # Thumb extend
        get_angle(landmark_list[17], landmark_list[18], landmark_list[20]) < 50  # Pinky folded
    )

def is_scroll(landmark_list):
    """Detects the 'OK' gesture and determines scroll direction based on hand height."""
    thumb_tip = landmark_list[4]
    index_finger_tip = landmark_list[8]
    middle_finger_tip = landmark_list[12]
    ring_finger_tip = landmark_list[16]
    pinky_finger_tip = landmark_list[20]

    # Check if thumb and index finger are touching
    thumb_index_distance = get_distance([thumb_tip, index_finger_tip])

    # Ensure middle, ring, and pinky fingers are extended
    fingers_extended = (
        middle_finger_tip[1] < landmark_list[9][1] and
        ring_finger_tip[1] < landmark_list[13][1] and
        pinky_finger_tip[1] < landmark_list[17][1]
    )

    if thumb_index_distance < 50 and fingers_extended:
        hand_y_position = index_finger_tip[1]  # Y-coordinate (higher = scroll up, lower = scroll down)
        return hand_y_position

    return None

def detect_gesture(frame, landmark_list, processed):
    if len(landmark_list) >= 21:

        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = get_distance([landmark_list[4], landmark_list[5]])

        if get_distance([landmark_list[4], landmark_list[5]]) < 50 and get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90:
            move_mouse(index_finger_tip)

        elif is_left_click(landmark_list, thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        elif is_right_click(landmark_list, thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        elif is_double_click(landmark_list, thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        elif is_voice_keyboard(landmark_list):
            speak("Voice Keyboard activated.")
            print("Activating Voice Keyboard...")
            text = capture_audio()
            type_text(text)
            cv2.putText(frame, "Voice Keyboard Activated", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        scroll_position = is_scroll(landmark_list)
        if scroll_position is not None:
            screen_mid = 0.5  # Midpoint of the screen (normalized)
            if scroll_position < screen_mid:
                pyautogui.scroll(30)  # Scroll up
                cv2.putText(frame, "Scrolling Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                pyautogui.scroll(-30)  # Scroll down
                cv2.putText(frame, "Scrolling Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)


        elif is_voice_activation(landmark_list, thumb_index_dist):
            speak("Voice assistant activated.")
            print("Activated Voice Assistant...")
            statement = takeCommand()
            if statement == "none":
                return

            if "good bye" in statement or "bye" in statement or "stop" in statement:
                speak('your personal assistant is shutting down,Good bye')
                print('your personal assistant is shutting down,Good bye')
                return

            if 'wikipedia' in statement:
                speak('Searching Wikipedia...')
                statement = statement.replace("wikipedia", "").strip()  # Remove 'wikipedia' and strip extra spaces
                if statement:
                    try:
                        results = wikipedia.summary(statement, sentences=3)
                        speak("According to Wikipedia")
                        print(results)
                        speak(results)
                    except wikipedia.exceptions.WikipediaException as e:
                        speak("Sorry, I couldn't find any information. Please try again later.")
                        print(f"Error: {e}")
                else:
                    speak("Please tell me what you'd like to search for on Wikipedia.")

            elif 'open youtube' in statement:
                webbrowser.open_new_tab("https://www.youtube.com")
                speak("youtube is open now")
                time.sleep(3)

            elif 'open' in statement and 'from desktop' in statement:
                open_application()

            elif 'open google' in statement:
                webbrowser.open_new_tab("https://www.google.com")
                speak("Google chrome is open now")
                time.sleep(3)

            elif 'open gmail' in statement:
                webbrowser.open_new_tab("gmail.com")
                speak("Google Mail open now")
                time.sleep(3)

            elif "weather" in statement:
                api_key = "bef821966f9a08f13d24c4457d87f88b"
                base_url = "https://api.openweathermap.org/data/2.5/weather?"
                speak("whats the city name")
                city_name = takeCommand()
                complete_url = base_url + "appid=" + api_key + "&q=" + city_name
                response = requests.get(complete_url)
                x = response.json()
                if x["cod"] != "404":
                    y = x["main"]
                    current_temperature = y["temp"]
                    current_humidiy = y["humidity"]
                    z = x["weather"]
                    weather_description = z[0]["description"]
                    speak(" Temperature in kelvin unit is " +
                          str(current_temperature) +
                          "\n humidity in percentage is " +
                          str(current_humidiy) +
                          "\n description  " +
                          str(weather_description))
                    print(" Temperature in kelvin unit = " +
                          str(current_temperature) +
                          "\n humidity (in percentage) = " +
                          str(current_humidiy) +
                          "\n description = " +
                          str(weather_description))

                else:
                    speak(" City Not Found ")

            elif 'time' in statement:
                strTime = datetime.datetime.now().strftime("%H:%M:%S")
                speak(f"the time is {strTime}")

            elif "who made you" in statement or "who created you" in statement or "who discovered you" in statement:
                speak("I was built by Tanish Nigam and Zaara Fatima")
                print("I was built by Tanish Nigam and Zaara Fatima")

            elif 'news' in statement:
                news = webbrowser.open_new_tab("https://timesofindia.indiatimes.com/home/headlines")
                speak('Here are some headlines from the Times of India,Happy reading')
                time.sleep(3)

            elif "camera" in statement or "take a photo" in statement:
                ec.capture(0, "robo camera", "img.jpg")

            elif 'where is' in statement:
                # Extract the location (after 'where')
                location = statement.replace("where", "").strip()
                if location:
                    # Open the location on Google Maps
                    speak(f"Searching for {location} on Google Maps")
                    webbrowser.open(f"https://www.google.com/maps?q={location}")
                else:
                    speak("Please specify the location you'd like to search.")

            elif 'search' in statement:
                # Replacing 'search' from the statement
                s = statement.replace('search', '')
                kit.search(s)  # This will search the modified query
                time.sleep(3)

            elif 'ask' in statement:
                speak('I can answer to computational and geographical questions. What would you like to ask now?')
                question = takeCommand()
                app_id = "RXAL2H-P99JR8H3PA"
                client = wolframalpha.Client(app_id)
                try:
                    res = client.query(question)
                    # Check if results exist before attempting to access them
                    if hasattr(res, 'results') and res.results:
                        answer = next(res.results).text
                        speak(answer)
                        print(answer)
                    else:
                        speak("Sorry, I couldn't find an answer to that question.")
                        print("No results found.")
                except Exception as e:
                    speak("Sorry, there was an error with your query.")
                    print(f"Error: {e}")

            elif "log off" in statement or "sign out" in statement:
                speak("Ok , your pc will log off in 10 sec make sure you exit from all applications")
                subprocess.call(["shutdown", "/l"])

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

def main():
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))

            detect_gesture(frame, landmark_list, processed)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()