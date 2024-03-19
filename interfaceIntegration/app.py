# Import necessary modules
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from io import BytesIO
from base64 import b64decode
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from flask_socketio import SocketIO

# Load your trained model
model = load_model('C:\\Users\\tanya\\Downloads\\my_model.h5')

# Configure the Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:\\Users\\tanya\\OneDrive\\Desktop\\classroom\\emotecare\\interfaceIntegration\\database\\emotion_data.db'
db = SQLAlchemy(app)

# Configure SocketIO
socketio = SocketIO(app)

# Define the EmotionData model
class EmotionData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_filename = db.Column(db.String(255))
    predicted_emotion = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Define the label2category dictionary (customize based on your model)
label2category = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Function to capture video from webcam, predict emotion, and store data in the database
def capture_emotion_webcam():
    try:
        # Open a connection to the webcam (assuming it's the first webcam, change if needed)
        cap = cv2.VideoCapture(0)

        # Create a VideoWriter object to save the video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_filename = 'static/video.avi'
        out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480), isColor=True)

        emotions = []  # List to store predicted emotions for each frame

        while True:
            # Capture a frame from the webcam
            ret, frame = cap.read()

            if not ret:
                break

            # Save the frame to the VideoWriter object
            out.write(frame)

            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Resize the frame to (48, 48) for consistency with the model input size
            resized_frame = cv2.resize(gray_frame, (48, 48))

            # Preprocess the frame for the model
            img = image.img_to_array(resized_frame)
            img = np.expand_dims(img, axis=0)
            img /= 255.0

            # Make predictions
            try:
                predicted_class_index = model.predict(img).argmax()
                predicted_category = label2category[predicted_class_index]
                emotions.append(predicted_category)
            except Exception as predict_err:
                print(f"Prediction error: {str(predict_err)}")

            # Display the frame
            cv2.imshow('Video Capture', frame)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and VideoWriter
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        if emotions:
            # Determine the most frequent predicted emotion
            predicted_emotion = max(set(emotions), key=emotions.count)

            # Store data in the database
            new_emotion_data = EmotionData(video_filename=video_filename, predicted_emotion=predicted_emotion)
            db.session.add(new_emotion_data)
            db.session.commit()

            # Return the filename and predicted emotion for further use
            return video_filename, predicted_emotion
        else:
            return None, None

    except Exception as err:
        print(f"Error: {str(err)}")
        return None, None

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture_webcam', methods=['POST'])
def capture_webcam():
    video_filename, predicted_emotion = capture_emotion_webcam()
    if video_filename and predicted_emotion:
        # Save the prediction to the database
        entry = EmotionData(video_filename=video_filename, predicted_emotion=predicted_emotion)
        db.session.add(entry)
        db.session.commit()

        # Emit a WebSocket event to update the frontend
        socketio.emit('update_result', {'video_filename': video_filename, 'predicted_emotion': predicted_emotion}, namespace='/test')
        return render_template('result.html', video_filename=video_filename, predicted_emotion=predicted_emotion)
    else:
        return "Error capturing video and predicting emotion."

@app.route('/reports')
def generate_reports():
    emotion_data = EmotionData.query.all()
    return render_template('reports.html', emotion_data=emotion_data)

# WebSocket event handler
@socketio.on('connect', namespace='/test')
def test_connect():
    print('Client connected')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    socketio.run(app, debug=True)

