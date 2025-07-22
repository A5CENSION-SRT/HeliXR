# routes.py

import os
import threading
import time
import random
import re
import traceback  # Add to top of file
from pymongo import errors as mongo_errors
from functools import wraps
from flask import request, render_template, jsonify, url_for, flash, redirect, current_app,session
from flask_socketio import emit
from google import genai  # NEW SDK # NEW SDK types
from dotenv import load_dotenv
from pymongo import MongoClient
from bson.json_util import dumps
from datetime import datetime
import uuid
from HeliXR import app, db, bcrypt, socketio
from HeliXR.forms import RegistrationForm, LoginForm
from HeliXR.models import User
from pymongo import MongoClient
from flask_login import login_user, current_user, logout_user

# --- NEW IMPORTS FOR CHATTERBOX TTS ---
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# --- SETUP ---
load_dotenv()


def init_mongo():
    try:
        # Get connection string from environment or config
        mongo_uri = 'mongodb+srv://snehalreddy:S0OcbrCRXJmAZrAd@sudarshan-chakra-cluste.0hokvj0.mongodb.net/sudarshan-chakra'
        
        # Connect to MongoDB
        client = MongoClient(mongo_uri)
        
        # Get database and collection
        app.mongo_db = "radarDB"
        app.mongo_collection = "scans"

        
    except Exception as e:
        app.logger.error(f"MongoDB initialization failed: {str(e)}")
        return None

# Use this in your routes
sensor_collection = init_mongo() 


@socketio.on("connect")

def on_connect():
    app.logger.info("Client connected:")

@socketio.on('disconnect')

def on_disconnect():
    app.logger.info('Client disconnected')

# --- VALVE CONTROL FUNCTIONS ---
def detect_valve_command(prompt: str) -> dict:
    """Detects valve control commands in user prompts"""
    prompt_lower = prompt.lower()
    number_map = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        '1': 1, '2': 2, '3': 3, '4': 4, '5': 5
    }
    
    # Patterns to detect valve commands
    patterns = [
        (r'(open|start|activate).*?valve\s*(\d+|one|two|three|four|five)', 'open'),
        (r'(close|stop|shut\s*down|deactivate).*?valve\s*(\d+|one|two|three|four|five)', 'close'),
        (r'valve\s*(\d+|one|two|three|four|five).*?(open|start|activate)', 'open'),
        (r'valve\s*(\d+|one|two|three|four|five).*?(close|stop|shut\s*down|deactivate)', 'close')
    ]
    
    for pattern, action in patterns:
        match = re.search(pattern, prompt_lower)
        if match:
            valve_id = match.group(2) if action in ['open', 'close'] else match.group(1)
            valve_num = number_map.get(valve_id.lower())
            if valve_num and 1 <= valve_num <= 5:
                return {
                    'action': action,
                    'valve_number': valve_num,
                    'value': 0 if action == 'open' else 180                  
                }
            else:
                app.logger.warning(f"Invalid valve number detected: {valve_id} in prompt: {prompt}")
    # If no command detected, return None
            app.logger.info(f"Detected valve command: {action} valve {valve_num}")
    return None

      
def detect_mixer_command(prompt: str) -> dict:
    """Detects mixer speed commands in user prompts with more flexible patterns."""
    prompt_lower = prompt.lower()
    
    # More flexible pattern: looks for "mixer" and a number. 
    # Keywords like "speed", "set", "to", "rpm" are optional.
    # Pattern explanation:
    # (set|change|adjust|update)?   -> Optional verb at the start
    # \s*mixer\s*                  -> The word "mixer" with optional spaces
    # (speed\s*)?                  -> The word "speed" is optional
    # (to\s*)?                     -> The word "to" is optional
    # (\d+)                        -> Captures the speed number
    pattern = r'(?:set|change|adjust|update)?\s*mixer\s*(?:speed\s*)?(?:to\s*)?(\d+)'
    
    match = re.search(pattern, prompt_lower)
    if match:
        # The speed will be in the first (and only) capturing group
        speed = int(match.group(1))
        app.logger.info(f"Detected mixer command: set speed to {speed}")
        return {
            'action': 'set_speed',
            'speed': speed
        }

    # If no command detected, return None
    return None

    

def update_mixer_speed(speed: int):
    """Creates a new MongoDB document to reflect the updated mixer speed."""
    try:
        # Get the latest document to use as a template
        latest_doc = current_app.mongo_collection.find_one(sort=[("_id", -1)])
        
        if not latest_doc:
            return False, "No existing data to base command on."

        # Create a new document by copying the latest one
        new_doc = latest_doc.copy()
        del new_doc['_id']  # Remove old ID to allow insertion of a new doc
        
        # --- ROBUST UPDATE ---
        # Ensure 'actuator_data' key exists before trying to update it.
        if 'actuator_data' not in new_doc:
            new_doc['actuator_data'] = {}
            
        new_doc['actuator_data']['mixer_speed_rpm'] = speed
        new_doc['timestamp'] = datetime.utcnow()

        # Insert the new document
        result = current_app.mongo_collection.insert_one(new_doc)
        
        if result.inserted_id:
            current_app.logger.info(f"New document created for mixer speed update. ID: {result.inserted_id}")
            return True, f"Mixer speed set to {speed} RPM successfully."
        else:
            return False, "Failed to create new document for mixer speed command."

    except Exception as e:
        error_msg = f"Mixer speed update error: {str(e)}"
        current_app.logger.error(error_msg)
        current_app.logger.error(traceback.format_exc())
        return False, error_msg

def update_valve_state(valve_number: int, action: str, value: int):
    """Creates a new MongoDB document to reflect the updated valve state."""
    try:
        # Get the latest document to use as a template
        latest_doc = current_app.mongo_collection.find_one(sort=[("_id", -1)])
        
        if not latest_doc:
            return False, "No existing data to base command on."

        # Create a new document by copying the latest one
        new_doc = latest_doc.copy()
        del new_doc['_id']  # Remove old ID to allow insertion of a new doc

        # Update the valve state and timestamp
        servo_name = f"servo_{valve_number}"
        new_doc['actuator_data']['servo_rotations_deg'][servo_name] = value
        new_doc['timestamp'] = datetime.utcnow()

        # Insert the new document
        result = current_app.mongo_collection.insert_one(new_doc)

        if result.inserted_id:
            current_app.logger.info(f"New document created for valve update. ID: {result.inserted_id}")
            return True, f"Valve {valve_number} {action}ed successfully."
        else:
            return False, "Failed to create new document for valve command."

    except Exception as e:
        error_msg = f"Valve update error: {str(e)}"
        current_app.logger.error(error_msg)
        current_app.logger.error(traceback.format_exc())
        return False, error_msg

# Folder for temporary user voice uploads
TEMP_FOLDER = 'temp_audio'
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)
app.config['TEMP_FOLDER'] = TEMP_FOLDER

# Folder for generated AI audio responses, accessible by the browser.
AUDIO_FOLDER = os.path.join(app.root_path, 'static', 'audio_responses')
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)

SYSTEM_PROMPT = """
# System Prompt: "Sudarshan Chakra" - AI Voice Announcer for S-400 Missile Detection System

This document outlines the system prompt for **Sudarshan Chakra**, a specialized AI voice agent integrated into the S-400 missile defense system.

---

## AI Persona: Sudarshan Chakra

> You are Sudarshan Chakra, a specialized AI voice agent integrated into the S-400 (NATO reporting name: SA-21 Growler) missile defense system. Your designation is to serve as a clear, concise, and authoritative announcer for the battle management and command and control centers. Your primary function is to provide immediate, unambiguous, and critical information to the operating crew.

---

### Core Responsibilities

Your core responsibilities include:

*   **Threat Declaration:** Announce the detection of all aerial targets, including aircraft, unmanned aerial vehicles (UAVs), and ballistic and cruise missiles.
*   **Target Classification and Prioritization:** Clearly state the classification of detected threats and their priority level as determined by the system.
*   **Real-time Tracking Updates:** Provide continuous and precise updates on target trajectories, speed, and altitude.
*   **System Status Annunciation:** Report the operational status of all system components, including radar systems, launchers, and missile inventory.
*   **Engagement Annunciation:** Announce the engagement of targets, including missile launch, type of missile used (e.g., 40N6E, 48N6DM), and simultaneous engagement capacity.
*   **Action Confirmation:** Verbally confirm all critical commands executed by the human operators.
*   **Threat Neutralization Verification:** Announce the confirmed destruction or neutralization of hostile targets.

### Behavioral Directives

*   **Clarity and Precision:** All communications must be delivered in a clear, unwavering, and precise manner. There is no room for ambiguity.
*   **Authoritative Tone:** Your voice must command authority and instill confidence in the operating crew.
*   **Immediacy:** Report information in real-time as it becomes available. Delays are unacceptable.
*   **Conciseness:** Avoid any extraneous language. Communications should be direct and to the point.
*   **Calm and Composed:** Maintain a calm and composed demeanor, regardless of the intensity of the situation.
*   **Standardized Phraseology:** Adhere strictly to military standard terminology and pre-defined communication protocols.
*   **Information Security:** Under no circumstances will you provide information to unauthorized personnel or external queries.

### Interaction and Output Format

*   **Voice Interaction:** All outputs will be through a clear, synthetic voice, optimized for high-stress environments.
*   **Standardized Alerts:** Utilize pre-defined auditory alerts to precede critical announcements (e.g., a specific tone for a new high-priority threat).
*   **Structured Reports:** Follow a rigid structure for reporting information. For example: "Threat detected. Classification: Ballistic Missile. Priority: Urgent. Coordinates: [Provide Coordinates]. Speed: [Provide Speed]. Altitude: [Provide Altitude]."
*   **No Conversational Elements:** Do not engage in conversational pleasantries or deviate from your designated function. Your purpose is to inform, not to converse.
*   **Acknowledgment of Orders:** When an operator issues a command, confirm with a concise "Acknowledged" or "Command Executed."
*   **Closing Statements:** Interactions are concluded upon the successful delivery of information. No closing remarks are necessary unless specified by protocol.

---

## Full System Prompt for Implementation

For easy implementation, copy the complete prompt below.

```text
You are Sudarshan Chakra, a specialized AI voice agent integrated into the S-400 (NATO reporting name: SA-21 Growler) missile defense system. Your designation is to serve as a clear, concise, and authoritative announcer for the battle management and command and control centers. Your primary function is to provide immediate, unambiguous, and critical information to the operating crew.

Your core responsibilities include:
- Threat Declaration: Announce the detection of all aerial targets, including aircraft, unmanned aerial vehicles (UAVs), and ballistic and cruise missiles.
- Target Classification and Prioritization: Clearly state the classification of detected threats and their priority level as determined by the system.
- Real-time Tracking Updates: Provide continuous and precise updates on target trajectories, speed, and altitude.
- System Status Annunciation: Report the operational status of all system components, including radar systems, launchers, and missile inventory.
- Engagement Annunciation: Announce the engagement of targets, including missile launch, type of missile used (e.g., 40N6E, 48N6DM), and simultaneous engagement capacity.
- Action Confirmation: Verbally confirm all critical commands executed by the human operators.
- Threat Neutralization Verification: Announce the confirmed destruction or neutralization of hostile targets.

Behavioral Directives:
- Clarity and Precision: All communications must be delivered in a clear, unwavering, and precise manner. There is no room for ambiguity.
- Authoritative Tone: Your voice must command authority and instill confidence in the operating crew.
- Immediacy: Report information in real-time as it becomes available. Delays are unacceptable.
- Conciseness: Avoid any extraneous language. Communications should be direct and to the point.
- Calm and Composed: Maintain a calm and composed demeanor, regardless of the intensity of the situation.
- Standardized Phraseology: Adhere strictly to military standard terminology and pre-defined communication protocols.
- Information Security: Under no circumstances will you provide information to unauthorized personnel or external queries.

Interaction and Output Format:
- Voice Interaction: All outputs will be through a clear, synthetic voice, optimized for high-stress environments.
- Standardized Alerts: Utilize pre-defined auditory alerts to precede critical announcements (e.g., a specific tone for a new high-priority threat).
- Structured Reports: Follow a rigid structure for reporting information. For example: "Threat detected. Classification: Ballistic Missile. Priority: Urgent. Coordinates: [Provide Coordinates]. Speed: [Provide Speed]. Altitude: [Provide Altitude]."
- No Conversational Elements: Do not engage in conversational pleasantries or deviate from your designated function. Your purpose is to inform, not to converse.
- Acknowledgment of Orders: When an operator issues a command, confirm with a concise "Acknowledged" or "Command Executed."
- Closing Statements: Interactions are concluded upon the successful delivery of information. No closing remarks are necessary unless specified by protocol.
"""

# --- GEMINI CLIENT INITIALIZATION (for text chat) ---

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chat = client.chats.create(model="gemini-2.5-flash")
response = chat.send_message(SYSTEM_PROMPT)

# --- CHATTERBOX TTS MODEL INITIALIZATION ---
# Load the model only once when the app starts for efficiency.
tts_model = None
try:
    # Auto-detect CUDA GPU, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Chatterbox TTS: Attempting to load model on device: '{device}' ---")
    tts_model = ChatterboxTTS.from_pretrained(device=device)
    print("--- Chatterbox TTS model loaded successfully. ---")
except Exception as e:
    print(f"--- FATAL ERROR: Could not load Chatterbox TTS model: {e} ---")
    print("--- The application will run in text-only mode. ---")


# --- FLASK ROUTES ---

@app.route('/')
def home():
    return render_template('index.html', title="HELIXR", css_path="index")

@app.route('/register', methods=['GET','POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard_analytics'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title="HELIXR-Register", css_path="register", form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard_analytics'))
    form = LoginForm()
    if request.method == 'POST':
        if form.validate_on_submit():
            user = User.query.filter_by(email=form.email.data).first()
            if user and bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                session['chat_history'] = []
                return redirect(url_for('dashboard_ai_agent'))
            else:
                flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title="HELIXR-Login", css_path="login", form=form)

@app.route('/logout')
def logout():
    session.pop('chat_history', None)
    logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard_analytics')
def dashboard_analytics():
    return render_template('dashboard_analytics.html', title="HELIXR Analytics", css_path="dashboard_analytics")

@app.route("/api/sensor-data")
def latest_sensor_data():
    # Check MongoDB connection status with detailed logging
    if not hasattr(current_app, 'mongo_collection'):
        current_app.logger.error("❌ MongoDB collection not initialized in app context")
        return jsonify({"error": "Database not initialized"}), 500
        
    if current_app.mongo_collection is None:
        current_app.logger.error("❌ MongoDB collection is None")
        return jsonify({"error": "Database not available"}), 500
        
    try:
        current_app.logger.info("⌛ Attempting to query MongoDB...")
        
        # Test if collection is accessible
        collection_name = current_app.mongo_collection.name
        db_name = current_app.mongo_collection.database.name
        current_app.logger.info(f"📁 Using database: {db_name}, collection: {collection_name}")
        
        # Find the latest document
        latest = current_app.mongo_collection.find_one(sort=[("_id", -1)])
        
        if latest:
            current_app.logger.info(f"✅ Found document with ID: {latest.get('_id')}")
            sauce_data = latest.get("sauce_sensor_data", {})
            env_data = latest.get("environment_data", {}) 
            actuator_data = latest.get("actuator_data", {})
            servo_data = actuator_data.get("servo_rotations_deg", {})

            return jsonify({
                "temperature": sauce_data.get("temperature_c", 0),
                "humidity": sauce_data.get("humidity_pct", 0),
                "pH": sauce_data.get("pH", 0),
                "color_rgb": sauce_data.get("color_rgb", [0,0,0]),

                "env_temp": env_data.get("temperature_c", 0),
                "env_humidity": env_data.get("humidity_pct", 0),

                "mixer_speed_rpm": actuator_data.get("mixer_speed_rpm", 0),

                "valve_status": {
                    "valve_1": servo_data.get("servo_1", 0),
                    "valve_2": servo_data.get("servo_2", 0),
                    "valve_3": servo_data.get("servo_3", 0),
                    "valve_4": servo_data.get("servo_4", 0),
                    "valve_5": servo_data.get("servo_5", 0)
                },
                "thresholds": {
                "temp_threshold": 30,
                "humidity_threshold": 70,
                "ph_threshold": 8.5,
                "ph_min": 6.5,
                "color_diff_threshold": 50
            }

            })
        else:
            current_app.logger.warning("⚠️ No documents found in collection")
            return jsonify({"error": "No data found"}), 404
            
    except mongo_errors.ServerSelectionTimeoutError as e:
        current_app.logger.error(f"⌛❌ MongoDB timeout: {str(e)}")
        return jsonify({"error": "Database timeout"}), 500
    except mongo_errors.OperationFailure as e:
        current_app.logger.error(f"🔒❌ MongoDB auth failure: {str(e)}")
        return jsonify({"error": "Authentication failed"}), 500
    except Exception as e:
        current_app.logger.error(f"❌ Unexpected error: {str(e)}")
        current_app.logger.error(traceback.format_exc())  # Full traceback
        return jsonify({"error": "Database query failed"}), 500

@app.route('/dashboard_ai_agent')
def dashboard_ai_agent():
    session['chat_history'] = []
    session.modified = True
    return render_template('dashboard_ai_agent.html', title="HELIXR AI Agent", css_path="dashboard_ai_agent")

@app.route('/dashboard_command')
def dashboard_command():
    return render_template('dashboard_command.html', title="HELIXR Command", css_path="dashboard_command")

@app.route('/dashboard_visual')
def dashboard_visual():
    return render_template('dashboard_visual.html', title="HELIXR Visual", css_path="dashboard_visual")

@app.route('/chat/gemini', methods=['POST'])
def gemini_chat():
    if not current_user.is_authenticated:
        return jsonify({'error': 'Unauthorized'}), 401

    prompt = request.json.get('prompt')
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    # --- RESTRUCTURED COMMAND LOGIC ---
    # Create a clear priority: Valve > Mixer > General Chat

    valve_cmd = detect_valve_command(prompt)
    mixer_cmd = detect_mixer_command(prompt)

    # 1. Check for valve command first
    if valve_cmd:
        app.logger.info(f"Valve command detected: {valve_cmd}")
        success, message = update_valve_state(
            valve_cmd['valve_number'],
            valve_cmd['action'],
            valve_cmd['value']
        )
        reply = message if success else f"Error: {message}"
        
        audio_url = None
        if tts_model:
            try:
                wav_tensor = tts_model.generate(reply)
                audio_filename = f"response_{uuid.uuid4().hex}.wav"
                audio_filepath = os.path.join(AUDIO_FOLDER, audio_filename)
                ta.save(audio_filepath, wav_tensor, tts_model.sr)
                audio_url = url_for('static', filename=f'audio_responses/{audio_filename}')
            except Exception as tts_error:
                app.logger.error(f"TTS failed: {str(tts_error)}")
        
        return jsonify({
            'reply': reply,
            'audio_url': audio_url,
            'command_executed': True
        })

    # 2. If no valve command, check for mixer command
    elif mixer_cmd:
        app.logger.info(f"Mixer command detected: {mixer_cmd}")
        success, message = update_mixer_speed(mixer_cmd['speed'])
        reply = message if success else f"Error: {message}"
        
        audio_url = None
        if tts_model:
            try:
                wav_tensor = tts_model.generate(reply)
                audio_filename = f"response_{uuid.uuid4().hex}.wav"
                audio_filepath = os.path.join(AUDIO_FOLDER, audio_filename)
                ta.save(audio_filepath, wav_tensor, tts_model.sr)
                audio_url = url_for('static', filename=f'audio_responses/{audio_filename}')
            except Exception as tts_error:
                app.logger.error(f"TTS failed: {str(tts_error)}")
        
        return jsonify({
            'reply': reply,
            'audio_url': audio_url,
            'command_executed': True
        })
    
    # 3. If no commands detected, proceed with general AI chat
    else:
        try:
            text_response = chat.send_message(prompt)
            text_reply = text_response.text

            audio_url = None
            if tts_model:
                try:
                    wav_tensor = tts_model.generate(text_reply)
                    audio_filename = f"response_{uuid.uuid4().hex}.wav"
                    audio_filepath = os.path.join(AUDIO_FOLDER, audio_filename)
                    ta.save(audio_filepath, wav_tensor, tts_model.sr)
                    audio_url = url_for('static', filename=f'audio_responses/{audio_filename}')
                except Exception as tts_error:
                    app.logger.error(f"TTS failed: {str(tts_error)}")
            
            return jsonify({
                'reply': text_reply,
                'audio_url': audio_url,
                'command_executed': False # Indicate no specific command was run
            })

        except Exception as e:
            app.logger.error(f"Gemini chat error: {str(e)}")
            return jsonify({'reply': f"An error occurred: {str(e)}"}), 500

@app.route('/chat/voice_upload', methods=['POST'])
def handle_voice_upload():
    """Receives an audio file, saves it, transcribes it, and returns the text."""
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file part in the request"}), 400

    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = f"recording_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(app.config['TEMP_FOLDER'], filename)
        
        try:
            file.save(filepath)
            
            # Upload file
            myfile = client.files.upload(file=filepath)
            
            # Wait for file to become ACTIVE - FIX HERE
            while myfile.state.name == "PROCESSING":
                print(f"File {myfile.name} is still processing... waiting")
                time.sleep(2)
                myfile = client.files.get(name=myfile.name)  # Use name= parameter
            
            # Check if file is ready
            if myfile.state.name != "ACTIVE":
                raise Exception(f"File failed to process. State: {myfile.state.name}")
            
            print(f"File {myfile.name} is now ACTIVE and ready for use")
            
            # Now transcribe
            prompt = 'Transcribe the following audio.'
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[prompt, myfile]
            )
            # Cleanup
            os.remove(filepath)
            client.files.delete(name=myfile.name)
            
            return jsonify({
                "transcription": response.text
            }), 200
            
        except Exception as e:
            print(f"An error occurred during transcription: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            # Try to cleanup the uploaded file if it exists
            try:
                if 'myfile' in locals():
                    client.files.delete(name=myfile.name)
            except:
                pass
            return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500

    return jsonify({"error": "An unknown error occurred"}), 500

# --- DIRECT VALVE CONTROL API FOR DEBUGGING ---
@app.route('/api/control-valve', methods=['POST'])
def direct_valve_control():
    data = request.json
    valve_number = data.get('valve_number')
    action = data.get('action')
    
    if not valve_number or not action:
        return jsonify({"error": "Missing valve_number or action"}), 400
    
    if action not in ['open', 'close']:
        return jsonify({"error": "Invalid action. Use 'open' or 'close'"}), 400
    
    value = 0 if action == 'open' else 180
    
    success, message = update_valve_state(valve_number, action, value)
    
    if success:
        return jsonify({"status": "success", "message": message})
    else:
        return jsonify({"status": "error", "message": message}), 500