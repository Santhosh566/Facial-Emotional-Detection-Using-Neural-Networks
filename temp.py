from flask import Flask, render_template, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, EqualTo
import psycopg2
import os
import base64
import uuid
import io
import numpy as np
from PIL import Image
from keras.models import load_model
import cv2
from flask import Flask, request, jsonify, Response
app = Flask(__name__,static_folder='static')

# Generate a random secret key
app.secret_key = os.urandom(24)

# PostgreSQL connection parameters
conn_params = {
    'dbname': 'Emotions',
    'user': 'postgres',
    'password': 'santhu',
    'host': 'localhost',
    'port': '5432'
}

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirmpassword = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

@app.route('/')
def indexuser():
    return render_template('indexuser.html')

@app.route('/loginuser', methods=['GET', 'POST'])
def loginuser():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        
        try:
            # Establish connection
            conn = psycopg2.connect(**conn_params)
            cursor = conn.cursor()

            # Execute SQL command
            cursor.execute('SELECT * FROM register WHERE username = %s AND password= %s', (username, password,))
            user = cursor.fetchone()

            if user:
                session['username'] = username
                #return redirect(url_for('indexuser'))
                return redirect(url_for('newpage'))
        except Exception as e:
            print("An error occurred:", e)
        finally:
            cursor.close()
            conn.close()
            
    return render_template('loginuser.html', form=form)
  # Redirect to login page if user not logged in
@app.route('/registeruser', methods=['GET', 'POST'])
def registeruser():
    form = RegisterForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        confirmpassword=form.confirmpassword.data
        
        try:
            # Establish connection
            conn = psycopg2.connect(**conn_params)
            cursor = conn.cursor()

            # Execute SQL command
            cursor.execute('INSERT INTO register (username, password,confirmpassword) VALUES (%s, %s, %s)', (username, password,confirmpassword))
            conn.commit()

            return redirect(url_for('loginuser'))
        except Exception as e:
            print("An error occurred:", e)
        finally:
            cursor.close()
            conn.close()
      
    return render_template('registeruser.html', form=form)

'''@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('indexuser'))'''
@app.route('/newpage')
def newpage():
    # Check if the user is logged in
    if 'username' in session:
        username = session['username']
        # Render the dashboard template with user information
        return render_template('newpage.html', username=username)
    else:
        # If the user is not logged in, redirect to the login page
        return redirect(url_for('loginuser'))
@app.route('/Preproc')
def Preproc():
    #os.system("python ems.py --mode display")
    os.system("final.py --mode display")
    return render_template('newpage.html')

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
# Load your trained Keras model
model = load_model('santhosh.keras')  # Load your trained emotion detection model

# Load your pre-trained generative model
def load_generative_model():
    # Replace this with code to load your generative model
    # Example:
    # model = load_model('path_to_your_generative_model')
    # return model
    return None

# Function to generate images based on emotion using the loaded model
def generate_image_from_emotion(model, emotion_name):
    # Replace this with code to generate images using your model
    # Example:
    # generated_image = model.generate_image(emotion_name)
    # return generated_image
    return None

# Route for the chatbot page
@app.route('/Chatbot', methods=['POST', 'GET'])
def chatbot():
    # Load the generative model (you can load it once when the server starts)
    model = load_generative_model()

    if request.method == 'POST':
        emotion_name = request.form['emotion_name']

        # Generate image based on the entered emotion using the model
        generated_image = generate_image_from_emotion(model, emotion_name)

        if generated_image is not None:
            # Convert the generated image to a byte stream
            img_byte_array = io.BytesIO()
            generated_image.save(img_byte_array, format='PNG')
            img_byte_array.seek(0)

            # Send the generated image as a response to display in the browser
            return send_file(img_byte_array, mimetype='image/png')

    # Render the chatbot HTML page
    return render_template('Chatbot.html')


model = load_model('santhosh.keras')

def predict_emotion(image):
    # Preprocess the image for the model
    image = cv2.resize(image, (48, 48))
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    image = image.astype('float32') / 255.0

    # Make a prediction using the model
    prediction = model.predict(image)
    emotion_label = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
    predicted_emotion = emotion_label[np.argmax(prediction)]

    return predicted_emotion

@app.route('/Dataloader', methods=['POST', 'GET'])
def dataloader():
    emotion_name = None
    
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('Dataloader.html', error_message='No image uploaded.')
        
        uploaded_image = request.files['image']
        if uploaded_image.filename == '':
            return render_template('Dataloader.html', error_message='No selected file.')
        
        # Read the uploaded image as a numpy array
        image_np = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

        # Perform emotion prediction
        emotion_name = predict_emotion(image_np)

    return render_template('Dataloader.html', emotion_name=emotion_name)

import logging
logging.basicConfig(level=logging.INFO)
@app.route('/logout', methods=['GET', 'POST'])
def logout():
    logging.info(f"Request Method: {request.method}")  # Log the request method
    if request.method == 'POST':
        session.clear()
        return redirect(url_for('loginuser'))  # Redirect to the login page
    else:
        return redirect(url_for('loginuser'))  # Redirect to the login page for GET requests
@app.route('/about')
def about():
    return render_template('about.html')
from flask_sqlalchemy import SQLAlchemy
# Configure SQLAlchemy settings
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# PostgreSQL connection parameters
conn_params = {
    'user': 'postgres',
    'password': 'santhu',
    'host': 'localhost',
    'port': '5432',
    'dbname': 'Emotions'
}

# Format the PostgreSQL connection URI
uri = 'postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}'.format(**conn_params)

# Set the SQLAlchemy database URI
app.config['SQLALCHEMY_DATABASE_URI'] = uri

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Define the ContactMessage model with an explicitly defined table name
class ContactMessage(db.Model):
    __tablename__ = 'contact_messages'  # Define your desired table name here

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f"ContactMessage(id={self.id}, name='{self.name}', email='{self.email}')"

# Route for handling the contact form
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        new_message = ContactMessage(name=name, email=email, message=message)
        db.session.add(new_message)
        db.session.commit()
        return 'Message sent successfully!'
    return render_template('contact.html')        
if __name__ == '__main__':
    app.run(debug=False)

