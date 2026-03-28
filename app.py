import os
import time
import json
import base64
import requests
import pandas as pd
import numpy as np
import math
import re
import csv
from math import radians, sin, cos, sqrt, asin
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, make_response
from flask_socketio import SocketIO, emit, join_room, leave_room, close_room
from dotenv import load_dotenv
from PIL import Image as PILImage
from fpdf import FPDF
import tensorflow as tf
from werkzeug.utils import secure_filename
import io
## --- UNVOICED MODEL SETUP & ENDPOINT (MOVED BELOW APP INIT) ---
# Place this after Flask app is initialized

import os
import time
import json
import requests
import pandas as pd
import numpy as np
import math
import re
import csv
from math import radians, sin, cos, sqrt, asin
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, make_response
from flask_socketio import SocketIO, emit, join_room, leave_room, close_room
from dotenv import load_dotenv
from PIL import Image
from fpdf import FPDF
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from werkzeug.utils import secure_filename
from agora_token_builder import RtcTokenBuilder

from pinecone import Pinecone
from langchain_groq import ChatGroq

load_dotenv()
 # Explicitly set template folder
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
app = Flask(__name__, template_folder=template_dir)
app.config['SECRET_KEY'] = 'secret!'

ASSETS_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))

UPLOAD_FOLDER = '/tmp/uploads'
OUTPUT_FOLDER = '/tmp/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- UNVOICED MODEL SETUP & ENDPOINT (MOVED BELOW APP INIT) ---
import cv2
import threading

UNVOICED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "Unvoiced-master", "trained_model_graph.pb")
UNVOICED_LABELS_PATH = os.path.join(os.path.dirname(__file__), "Unvoiced-master", "training_set_labels.txt")

# Load labels
try:
    with tf.io.gfile.GFile(UNVOICED_LABELS_PATH, 'r') as f:
        UNVOICED_LABELS = [line.strip() for line in f.readlines()]
except Exception as e:
    print(f"[Unvoiced] Failed to load labels: {e}")
    UNVOICED_LABELS = []

# Load graph
def load_unvoiced_graph():
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(UNVOICED_MODEL_PATH, 'rb') as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    return graph

UNVOICED_GRAPH = load_unvoiced_graph()
UNVOICED_SESSION = tf.compat.v1.Session(graph=UNVOICED_GRAPH)
UNVOICED_LOCK = threading.Lock()

@app.route('/predict_unvoiced', methods=['POST'])
def predict_unvoiced():
    if 'frame' not in request.files:
        return jsonify({'success': False, 'error': 'No frame uploaded'}), 400
    file = request.files['frame']
    try:
        # Read image from file
        in_bytes = file.read()
        npimg = np.frombuffer(in_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        # Preprocess: resize to 200x200, grayscale, normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (200, 200))
        img = img.astype(np.float32) / 255.0
        img = img.reshape(1, 200, 200, 1)
        # Predict
        with UNVOICED_LOCK:
            input_tensor = UNVOICED_GRAPH.get_tensor_by_name('input_1:0')
            output_tensor = UNVOICED_GRAPH.get_tensor_by_name('dense_1/Softmax:0')
            preds = UNVOICED_SESSION.run(output_tensor, feed_dict={input_tensor: img})
        idx = int(np.argmax(preds[0]))
        letter = UNVOICED_LABELS[idx] if UNVOICED_LABELS and idx < len(UNVOICED_LABELS) else 'unknown'
        confidence = float(preds[0][idx])
        return jsonify({'success': True, 'letter': letter, 'confidence': confidence})
    except Exception as e:
        print(f"[Unvoiced] Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# --- CONFIGURATION KEYS ---
API_NINJAS_KEY = os.getenv("API_NINJAS_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY_NEW")
PINECONE_KEY = "pcsk_6dPaFR_TvSs7euNhKjG8tmiaKoQvJ4xZ7AoGwoZkjPHmxnTae3ELTubUbz4pxmh4toR48n"
HIDDEN_CSV_URL = os.getenv("HIDDEN_CSV_URL1")
LOCALITY_BIN_ID = os.getenv("LOCALITY_BIN_ID")
JSONBIN_KEY = os.getenv("JSONBIN_MASTER_KEY")

# --- EMERGENCY / TWILIO (OPTIONAL) ---
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "").strip()
TWILIO_TO_NUMBER = os.getenv("TWILIO_TO_NUMBER", "").strip()

# --- FIREBASE (for emergency contacts) ---
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "").strip()
FIREBASE_API_KEY    = os.getenv("FIREBASE_API_KEY", "").strip()

# Comma-separated fallback list (used when Firebase is not configured / project ID is wrong)
EMERGENCY_CONTACTS_ENV = [
    n.strip() for n in os.getenv("EMERGENCY_CONTACTS", "").split(",") if n.strip()
]

# --- AGORA CONFIGURATION ---
AGORA_APP_ID = "2d12c3a1fb854e35a0f9df5d6264258c"
AGORA_APP_CERTIFICATE = "6453060325ef434784df7f75703cf9ba"

# --- AZURE CONFIGURATION ---
# Prefer env vars so you can supply your own Azure Speech subscription key.
# If missing, /get-token will return a helpful 500 error.
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_REGION = os.getenv("AZURE_REGION", "centralindia")
GLOSSARY_FILE = "glossary_expanded.csv"

# Ping timeout: 5 seconds
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=60, ping_interval=25)

# --- GLOBAL STATE ---
# Structure: { 'room_id': { 'doctor_sid': 'xyz', 'patients': ['abc'] } }
active_rooms = {}
print("🧠 Loading Cancer Model...")

def build_cancer_model():
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(4, activation="softmax") # 4 Classes
    ])
    return model

# --- NEW: Function to Download Model from Dropbox ---
def download_model_if_missing(filename, url):
    if not os.path.exists(filename):
        print(f"⬇️ Model file '{filename}' not found. Downloading from Dropbox...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Check for errors
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Download complete.")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            return False
    return True

# --- CONFIGURATION ---
MODEL_FILENAME = "chest_cancer_classifier.h5"
# 👇 PASTE YOUR DROPBOX LINK HERE AND ENSURE IT ENDS WITH ?dl=1
DROPBOX_URL = "https://www.dropbox.com/scl/fi/8t52vx5wdo43oftzdv9kj/chest_cancer_classifier.h5?rlkey=hhuisl6pr1zz8q1u46i3rekw5&st=w878b8m2&dl=1"

# 1. Build Architecture
cancer_model = build_cancer_model()

# 2. Download File (if needed)
download_model_if_missing(MODEL_FILENAME, DROPBOX_URL)



# 3. Load Weights
try:
    cancer_model.load_weights(MODEL_FILENAME)
    print("✅ Cancer Model Weights Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading cancer model weights: {e}")

CLASS_NAMES = ["Adenocarcinoma", "Large_cell", "Normal", "Squamous_cell"]

# --- INITIALIZE AI CLIENTS ---
# 1. Pinecone (The Database)
try:

    print(f"DEBUG: Using Pinecone API Key: {PINECONE_KEY}")
    pc = Pinecone(api_key=PINECONE_KEY)
    try:
        rag_index = pc.Index("newmedical")
        print("✅ Pinecone 'newmedical' index connected.")
    except Exception as e2:
        print(f"WARNING: Pinecone index connection failed: {e2}")
        rag_index = None
except Exception as e:
    print(f"WARNING: Pinecone client init failed: {e}")
    rag_index = None

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    api_key=GROQ_API_KEY
)

# --- AGORA TOKEN GENERATOR ---
def generate_agora_token(channel_name, uid):
    expiration_time_in_seconds = 3600 * 24 
    current_timestamp = int(time.time())
    privilege_expired_ts = current_timestamp + expiration_time_in_seconds
    
    if AGORA_APP_CERTIFICATE == "PASTE_YOUR_APP_CERTIFICATE_HERE":
        print("⚠️ WARNING: Agora App Certificate is missing. Video might fail.")

    token = RtcTokenBuilder.buildTokenWithUid(
        AGORA_APP_ID, AGORA_APP_CERTIFICATE, channel_name, uid, 1, privilege_expired_ts
    )
    return token

def load_glossary():
    structured_vocab = []
    if not os.path.exists(GLOSSARY_FILE): return []
    try:
        with open(GLOSSARY_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue 
                entry = {'english': '', 'tamil_synonyms': []}
                if len(row) >= 1:
                    raw_synonyms = row[0].split('|')
                    entry['tamil_synonyms'] = [s.strip() for s in raw_synonyms if s.strip()]
                if len(row) > 1:
                    entry['english'] = row[1].strip()
                if entry['english'] and entry['tamil_synonyms']:
                    structured_vocab.append(entry)
    except: return []
    return structured_vocab

def broadcast_room_status(room):
    if room not in active_rooms: return
    info = active_rooms[room]
    has_doctor = info['doctor_sid'] is not None
    has_patient = len(info['patients']) > 0
    emit('status_update', {'has_doctor': has_doctor, 'has_patient': has_patient}, room=room)

# --- GEO & HOSPITAL LOGIC ---

def haversine(lat1, lon1, lat2, lon2):
    """Calculates the great circle distance between two points."""
    if any(x == 0 for x in [lat1, lon1, lat2, lon2]): return 0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 6371 * 2 * asin(sqrt(a))

def fetch_osrm_data(lat1, lon1, lat2, lon2):
    """Fetches real driving route geometry and distance."""
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            d = r.json()
            if "routes" in d and len(d["routes"]) > 0:
                route = d["routes"][0]
                geometry = route["geometry"]
                distance_km = route["distance"] / 1000
                duration_mins = route["duration"] / 60  
                return geometry, distance_km, duration_mins
    except: pass
    return None, 0, 0

def fetch_jsonbin(bin_id):
    """Fetches locality data from JSONBin.io."""
    if not bin_id or not JSONBIN_KEY: return {}
    url = f"https://api.jsonbin.io/v3/b/{bin_id}/latest"
    headers = {"X-Master-Key": JSONBIN_KEY}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            return resp.json().get("record", {})
    except: pass
    return {}


def get_ai_coordinates(place_name):
    """
    Asks the LLM for the latitude and longitude of a specific place in Chennai
    using a single string prompt.
    """
    prompt = f"""
    You are a Geospatial Assistant for Chennai, India.
    User will give you a place name (company, college, landmark).
    You must return EXACTLY the Latitude and Longitude in this format:
    lat,lon
    
    Example:
    User: "Tidel Park"
    Output: 12.9892, 80.2483
    
    User: "VIT Chennai"
    Output: 12.8406, 80.1534
    
    If you absolutely do not know, return: 0,0
    DO NOT output any other text. Just the numbers.
    
    User Request: {place_name}
    """
    
    try:
        # Note: Using 'llm' because that's what we named the Groq client in app.py
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        match = re.search(r"(-?\d+\.\d+),\s*(-?\d+\.\d+)", content)
        if match:
            return (float(match.group(1)), float(match.group(2)))
        return (0.0, 0.0)
    except:
        return (0.0, 0.0)
    
# --- NEW LLM TRIAGE FUNCTION ---
def analyze_severity_with_llm(symptoms_text):
    """
    Uses Groq/Llama-3 to classify severity and specialty.
    """
    prompt = f"""
    You are an Emergency Triage AI. Analyze the patient's symptoms and classify them strictly into JSON format.

    INPUT SYMPTOMS: "{symptoms_text}"

    RULES:
    1. Determine the 'specialty' (e.g., Cardiology, Orthopedics, General Medicine, Neurology, Ophthalmology).
    2. Assign a 'severity_score' from 1 (mild) to 10 (life-threatening).
    3. Assign a 'triage_color':
       - RED: Immediate emergency (Stroke, Heart Attack, Severe Trauma, Breathlessness).
       - YELLOW: Urgent but stable (Fractures, High Fever, Persistent Pain).
       - GREEN: Non-urgent (Mild Cold, Skin Rash, Routine).
    4. Set 'is_emergency': true if Red, else false.

    OUTPUT FORMAT (Pure JSON only, no markdown):
    {{
        "score": <int>,
        "color": "<string>",
        "specialty": "<string>",
        "is_emergency": <bool>,
        "alert_msg": "<Short warning message>"
    }}
    """
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        # Clean markdown if present
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"LLM Triage Error: {e}")
        return {"score": 5, "color": "YELLOW", "specialty": "General Medicine", "alert_msg": "Consult a doctor."}

class ChennaiGeoEngine:
    def __init__(self):
        self.coords_db = fetch_jsonbin(LOCALITY_BIN_ID)
        if not self.coords_db: self.coords_db = {}

    def get_route_geometry(self, lat1, lon1, lat2, lon2):
        geom, distance, duration = fetch_osrm_data(lat1, lon1, lat2, lon2)
        if distance > 0:
            return geom, distance, duration
        return None, haversine(lat1, lon1, lat2, lon2), 3*haversine(lat1, lon1, lat2, lon2)

    def resolve_location(self, query):
        """Resolves text location to Lat/Lon with LLM Fallback."""
        clean = str(query).lower().strip()
        
        # 1. Check Internal Database (JSONBin)
        for key in self.coords_db:
            if key in clean:
                val = self.coords_db[key]
                return (float(val[0]), float(val[1])), "Database"

        # 2. Check LLM (The logic you requested)
        try:
            print(f"DEBUG: Asking AI for coordinates of '{query}'...")
            llm_lat_lon = get_ai_coordinates(clean)
            if llm_lat_lon and llm_lat_lon != (0.0, 0.0):
                return llm_lat_lon, "LLM Generated"
        except Exception as e:
            print(f"LLM Geo Error: {e}")

        # 3. Check Online (Nominatim Fallback)
        try:
            url = "https://nominatim.openstreetmap.org/search"
            resp = requests.get(url, params={'q': f"{query}, Chennai", 'format': 'json', 'limit': 1}, headers={'User-Agent': 'ReloApp/1.0'}, timeout=2)
            if resp.status_code == 200 and resp.json():
                d = resp.json()[0]
                return (float(d['lat']), float(d['lon'])), "Online"
        except: pass
        
        return (0.0, 0.0), "Unknown"

    def generate_map_html_string(self, user_coords, route_geometry, hospitals_list):
        """Generates a Leaflet Map HTML string."""
        user_data = []
        if user_coords and user_coords[0] != 0:
            user_data.append({
                "lat": user_coords[0],
                "lon": user_coords[1],
                "popup": "<b>Your Location</b>"
            })

        hosp_data = []
        if hospitals_list:
            for h in hospitals_list:
                time_info = h.get('time', 'N/A')
                hosp_data.append({
                    "lat": h['lat'],
                    "lon": h['lon'],
                    "popup": f"<b>{h['name']}</b><br>{h['dist']} km<br>{time_info}"
                })

        route_json = json.dumps(route_geometry) if route_geometry else "null"

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8" />
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <style>
                body {{ margin: 0; padding: 0; }}
                #map {{ width: 100%; height: 500px; border-radius: 12px; }}
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script>
                var map = L.map('map').setView([13.05, 80.23], 11);
                L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png', {{
                    attribution: '© OpenStreetMap © CARTO', maxZoom: 19
                }}).addTo(map);

                var user_loc = {json.dumps(user_data)};
                var hospitals = {json.dumps(hosp_data)};
                var route = {route_json};
                var bounds = L.latLngBounds();

                // Plot User
                user_loc.forEach(function(w) {{
                    var icon = L.icon({{
                        iconUrl: 'https://cdn-icons-png.flaticon.com/512/3005/3005357.png',
                        iconSize: [32, 32], iconAnchor: [16, 32], popupAnchor: [0, -32]
                    }});
                    L.marker([w.lat, w.lon], {{icon: icon}}).addTo(map).bindPopup(w.popup);
                    bounds.extend([w.lat, w.lon]);
                }});

                // Plot Hospitals
                hospitals.forEach(function(h) {{
                    var icon = L.icon({{
                        iconUrl: 'https://cdn-icons-png.flaticon.com/512/4320/4320371.png',
                        iconSize: [24, 24], iconAnchor: [12, 24], popupAnchor: [0, -24]
                    }});
                    L.marker([h.lat, h.lon], {{icon: icon}}).addTo(map).bindPopup(h.popup);
                    bounds.extend([h.lat, h.lon]);
                }});

                // Plot Route
                if (route) {{
                    L.geoJSON(route, {{ style: {{ color: '#3388ff', weight: 5, opacity: 0.7 }} }}).addTo(map);
                }}

                if (user_loc.length > 0 || hospitals.length > 0) {{
                    map.fitBounds(bounds, {{padding: [50, 50]}});
                }}
            </script>
        </body>
        </html>
        """
        return html

class HospitalEngine:
    def __init__(self):
        self.csv_url = HIDDEN_CSV_URL
        self.data = None
        if self.csv_url:
            try:
                print(f"DEBUG: Attempting to load CSV from {self.csv_url}...")
                # Added error_bad_lines=False (for older pandas) or on_bad_lines='skip'
                try:
                    self.data = pd.read_csv(self.csv_url, on_bad_lines='skip')
                except TypeError:
                    self.data = pd.read_csv(self.csv_url, error_bad_lines=False)
                
                # Clean column names
                self.data.columns = [c.strip().lower() for c in self.data.columns]
                print(f"DEBUG: Columns found: {self.data.columns.tolist()}")

                # --- MANUAL MAPPING FOR YOUR CSV ---
                self.name_col = 'name'
                self.addr_col = 'full_address'
                self.lat_col = 'latitude'
                self.lon_col = 'longitude'
                self.spec_col = 'Specialty'
                
                # Verify columns exist
                if self.lat_col not in self.data.columns or self.lon_col not in self.data.columns:
                    raise ValueError(f"Missing lat/lon columns. Found: {self.data.columns.tolist()}")

                # Clean Data
                self.data = self.data.dropna(subset=[self.lat_col, self.lon_col])
                self.data[self.lat_col] = self.data[self.lat_col].astype(float)
                self.data[self.lon_col] = self.data[self.lon_col].astype(float)
                print(f"DEBUG: Successfully loaded {len(self.data)} hospitals from CSV.")

            except Exception as e:
                print(f"❌ ERROR Loading CSV: {e}")
                self.data = None # Trigger fallback below

        # 2. Safety Fallback (If CSV fails, use this so app doesn't break)
        if self.data is None or self.data.empty:
            print("⚠️ WARNING: Using Fallback (Hardcoded) Hospital Data.")
            self.data = pd.DataFrame({
                'name': ['Apollo Main Hospital (Fallback)', 'MIOT International', 'Fortis Malar', 'Billroth Hospital', 'Sri Ramachandra Medical Centre'], 
                'full_address': ['Greams Lane, Chennai', 'Manapakkam, Chennai', 'Gandhi Nagar, Adyar', 'Shenoy Nagar, Chennai', 'Porur, Chennai'],
                'latitude': [13.064, 13.027, 13.006, 13.078, 13.033], 
                'longitude': [80.250, 80.180, 80.257, 80.228, 80.158]
            })
            self.name_col = 'name'
            self.addr_col = 'full_address'
            self.lat_col = 'latitude'
            self.lon_col = 'longitude'

    def find_nearest_n(self, lat, lon, n=5, required_specialty=None):
        if self.data is None or lat == 0: return []
    
        # Work on the full dataset
        df_full = self.data.copy()
        
        # 1. PRE-CALCULATE DISTANCE FOR EVERYONE
        # We need this for sorting later, but we won't filter by it yet.
        df_full["approx_km"] = df_full.apply(lambda r: haversine(lat, lon, r[self.lat_col], r[self.lon_col]), axis=1)
        
        # 2. DEFINE COLUMNS
        last_col_series = df_full.iloc[:, -1].astype(str) # Specialty Column
        name_series = df_full[self.name_col].astype(str)  # Name Column
        
        # 3. DEFINE LISTS
        specialists_df = pd.DataFrame()
        backups_df = pd.DataFrame()

        if required_specialty and required_specialty.lower() != "general medicine":
            target = required_specialty.lower()
            
            # --- A. DEFINE SEARCH TERMS ---
            if "ortho" in target:
                spec_regex = "ortho|bone|fracture|kattu|joint|spine"
            elif "psych" in target:
                spec_regex = "psych|mental|mind"
            elif "cardio" in target:
                spec_regex = "cardio|heart"
            else:
                spec_regex = target

            # --- B. FIND SPECIALISTS (GLOBAL SCAN) ---
            # checking the LAST column for the keyword (e.g. "Ortho")
            mask_specialist = last_col_series.str.contains(spec_regex, case=False, na=False)
            
            specialists_df = df_full[mask_specialist].copy()
            specialists_df['match_type'] = "Specialist"
            specialists_df['match_priority'] = 0 # Top Priority
            
            # Sort specialists by distance immediately
            specialists_df = specialists_df.sort_values("approx_km")

            # --- C. FIND BACKUPS (GENERAL HOSPITALS) ---
            # We search the REST of the data (exclude rows we already found in specialists_df)
            
            # 1. Must contain "General" or "Multi"
            backup_regex = "general|multi|family medicine|internal medicine"
            mask_general = last_col_series.str.contains(backup_regex, case=False, na=False)
            
            # 2. STRICT BLACKLIST (Kill Vasan Eye Care)
            banned_words = "eye|ophthal|opthal|vision|netra|retina|dental|tooth|teeth|skin|derma|hair|opticals|lasik"
            mask_banned_name = name_series.str.contains(banned_words, case=False, na=False)
            mask_banned_spec = last_col_series.str.contains(banned_words, case=False, na=False)
            mask_is_banned = mask_banned_name | mask_banned_spec

            # 3. Exclude already found specialists
            mask_not_specialist = ~df_full.index.isin(specialists_df.index)

            # Apply Logic
            backups_df = df_full[mask_general & (~mask_is_banned) & mask_not_specialist].copy()
            backups_df['match_type'] = "General Hospital (Backup)"
            backups_df['match_priority'] = 1 # Lower Priority
            
            # Sort backups by distance
            backups_df = backups_df.sort_values("approx_km")

            # --- D. COMBINE ---
            # Put Specialists ON TOP, Backups BELOW
            candidates = pd.concat([specialists_df, backups_df])
            
        else:
            # General Medicine Logic
            banned_words = "eye|ophthal|opthal|vision|netra|retina|dental|tooth|teeth|skin|derma|hair"
            mask_banned = last_col_series.str.contains(banned_words, case=False, na=False) | \
                          name_series.str.contains(banned_words, case=False, na=False)
            
            candidates = df_full[~mask_banned].sort_values("approx_km")
            candidates['match_type'] = "Nearest Facility"
            candidates['match_priority'] = 0

        # --- E. FINAL SLICE & API CALL ---
        # Now we have a list: [Ortho 10km, Ortho 50km, General 1km, General 2km...]
        # We take the top N * 2 to send to API
        candidates = candidates.head(n * 2).head(25)

        if candidates.empty: return []

        results = []
        try:
            # OSRM API Call
            coords_str = f"{lon},{lat};" + ";".join([f"{r[self.lon_col]},{r[self.lat_col]}" for _, r in candidates.iterrows()])
            dest_indices = ";".join([str(i+1) for i in range(len(candidates))])
            
            url = f"http://router.project-osrm.org/table/v1/driving/{coords_str}?sources=0&destinations={dest_indices}&annotations=duration,distance"
            resp = requests.get(url, timeout=5)
            
            if resp.status_code == 200:
                data = resp.json()
                durations = data["durations"][0] 
                distances = data["distances"][0] 
            
                for i, (_, row) in enumerate(candidates.iterrows()):
                    if durations[i] is None: continue 

                    # FINAL GATEKEEPER: Double check name one last time
                    name_check = str(row[self.name_col]).lower()
                    if any(x in name_check for x in ["eye", "ophthal", "dental", "skin", "vision"]):
                         continue 

                    results.append({
                        "name": row[self.name_col],
                        "address": str(row[self.addr_col])[:40] + "...",
                        "dist": round(distances[i] / 1000, 2), 
                        "time": f"{int(round(durations[i] / 60, 0))} mins", 
                        "lat": row[self.lat_col],
                        "lon": row[self.lon_col],
                        "match_type": row['match_type'],
                        "match_priority": row['match_priority']
                    })
        except Exception as e:
             # Fallback
             for _, row in candidates.iterrows():
                name_check = str(row[self.name_col]).lower()
                if any(x in name_check for x in ["eye", "ophthal", "dental", "skin", "vision"]):
                         continue
                         
                results.append({
                    "name": row[self.name_col],
                    "address": str(row[self.addr_col])[:40] + "...",
                    "dist": round(row["approx_km"], 2),
                    "time": f"{int(row['approx_km']*3)} mins (Est)",
                    "lat": row[self.lat_col],
                    "lon": row[self.lon_col],
                    "match_type": row['match_type'] + " (Approx)",
                    "match_priority": row['match_priority']
                })

        # IMPORTANT: Sort by Priority (0 vs 1) first, then Driving Distance
        results.sort(key=lambda x: (x['match_priority'], x['dist']))
        
        return results[:n]

    

# Initialize Engines
geo_engine = ChennaiGeoEngine()
hospital_engine = HospitalEngine()

# ==========================================
#  PART 2: PINECONE RAG LOGIC (The Doctor)
# ==========================================

def get_medical_rag_prediction(symptoms):
    """
    GROQ-ONLY VERSION:
    Takes symptoms → asks LLM → returns top 5 diseases.
    """
    try:
        print(f"DEBUG: Sending symptoms to Groq: {symptoms}")

        prompt = f"""
You are a highly accurate medical diagnostic AI.

TASK:
Based ONLY on the symptoms provided, predict the TOP 5 most likely diseases.

RULES:
- Output ONLY valid JSON (no text before or after)
- Format:
[
    {{"disease": "Disease Name", "description": "Short explanation"}},
    ...
]
- Always return EXACTLY 5 diseases
- Keep descriptions short and medically relevant

SYMPTOMS:
{symptoms}
"""

        response = llm.invoke(prompt)
        content = response.content.strip()

        print("DEBUG Groq Raw:", content)

        # Extract JSON safely
        start = content.find('[')
        end = content.rfind(']') + 1

        if start == -1 or end == 0:
            raise ValueError("Invalid JSON from Groq")

        clean_json = content[start:end]

        return json.loads(clean_json)

    except Exception as e:
        print(f"GROQ ERROR: {e}")

        return [
            {"disease": "Migraine", "description": "Common headache cause."},
            {"disease": "Tension Headache", "description": "Stress-related pain."},
            {"disease": "Sinusitis", "description": "Sinus inflammation."},
            {"disease": "Flu", "description": "Viral infection."},
            {"disease": "Dehydration", "description": "Lack of fluids."}
        ]

@app.route('/')
def home():
    index_path = os.path.join(template_dir, 'index.html')
    if not os.path.exists(index_path):
        return "❌ index.html not found in templates folder", 404
    resp = make_response(render_template('index.html'))
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


@app.route('/predict_cancer', methods=['POST'])
def predict_cancer():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Preprocess Image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            preds = cancer_model.predict(img_array)
            class_index = np.argmax(preds[0])
            confidence = float(preds[0][class_index] * 100)
            predicted_class = CLASS_NAMES[class_index]

            status = "NON-CANCEROUS" if predicted_class == "Normal" else "CANCEROUS"
            
            # Simple advice based on result
            advice = "Consult an Oncologist immediately." if status == "CANCEROUS" else "Routine checkups recommended."

            return jsonify({
                'success': True,
                'status': status,
                'class': predicted_class,
                'confidence': round(confidence, 2),
                'advice': advice
            })

        except Exception as e:
            print(f"Prediction Error: {e}")
            return jsonify({'success': False, 'error': str(e)})

    return jsonify({'success': False, 'error': 'Unknown error'})



@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    """
    New Workflow:
    1. Receives Symptoms.
    2. Calls Pinecone RAG to get Disease Predictions.
    3. Returns JSON.
    """
    data = request.json
    symptoms = data.get('symptoms', '')
    
    if not symptoms:
        return jsonify({"success": False, "message": "No symptoms provided."})

    # Call RAG
    predictions = get_medical_rag_prediction(symptoms)
    triage_info = analyze_severity_with_llm(symptoms)
    return jsonify({
        "success": True,
        "predictions": predictions,
        "triage": triage_info
    })

@app.route('/locate_hospitals', methods=['POST'])
def locate_hospitals():
    """
    New Workflow:
    1. Receives User Location (Text like 'Adyar' OR Lat/Lon from Browser).
    2. Geocodes it.
    3. Finds Nearest Hospitals (using CSV data + OSRM).
    4. Generates HTML Map with Route.
    """
    data = request.json
    location_text = data.get('location_text', '')
    lat_in = data.get('Latitude')
    lon_in = data.get('Longitude')
    
    user_coords = (0, 0)

    # 1. Determine User Coordinates
    if lat_in and lon_in:
        user_coords = (float(lat_in), float(lon_in))
    elif location_text:
        user_coords, _ = geo_engine.resolve_location(location_text)
    
    if user_coords == (0, 0):
        return jsonify({"success": False, "message": "Could not determine location."})

    # 2. Find Hospitals
    nearest_hospitals = hospital_engine.find_nearest_n(user_coords[0], user_coords[1], n=5)
    
    # 3. Get Route to Closest Hospital
    route_geom = None
    if nearest_hospitals:
        closest = nearest_hospitals[0]
        route_geom, _, _ = geo_engine.get_route_geometry(
            user_coords[0], user_coords[1], 
            closest['lat'], closest['lon']
        )
    
    # 4. Generate Map
    map_html = geo_engine.generate_map_html_string(user_coords, route_geom, nearest_hospitals)
    
    return jsonify({
        "success": True,
        "user_coords": user_coords,
        "nearest_hospitals": nearest_hospitals,
        "map_html": map_html
    })

# --- UTILS (OCR & PDF) ---

def optimize_image(image_path, max_size_kb=200):
    file_size = os.path.getsize(image_path) / 1024
    if file_size <= max_size_kb: return image_path
    img = Image.open(image_path)
    if img.mode in ("RGBA", "P"): img = img.convert("RGB")
    if img.width > 1024:
        ratio = 1024 / float(img.width)
        img = img.resize((1024, int(float(img.height) * ratio)), Image.Resampling.LANCZOS)
    optimized_filename = f"compressed_{os.path.basename(image_path)}"
    optimized_path = os.path.join(UPLOAD_FOLDER, optimized_filename)
    img.save(optimized_path, optimize=True, quality=40)
    return optimized_path

def get_ocr_text(image_path):
    url = "https://api.api-ninjas.com/v1/imagetotext"
    with open(image_path, "rb") as img_file:
        files = {"image": img_file}
        headers = {"X-Api-Key": API_NINJAS_KEY}
        resp = requests.post(url, files=files, headers=headers)
        if resp.status_code != 200: raise Exception(f"OCR Error: {resp.text}")
        data = resp.json()
        return " ".join([block["text"].strip() for block in data if block["text"].strip()])

def generate_pdf(text_content, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Medical Report Analysis", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", size=11)
    safe_text = text_content.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, safe_text)
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    pdf.output(filepath)
    return filepath

@app.route('/analyze', methods=['POST'])
def analyze():
    # Helper to clean up old files
    now = time.time()
    for f in os.listdir(UPLOAD_FOLDER):
        if os.stat(os.path.join(UPLOAD_FOLDER,f)).st_mtime < now - 600: os.remove(os.path.join(UPLOAD_FOLDER,f))

    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No file"}), 400

    filename = f"{int(time.time())}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        final_path = optimize_image(filepath)
        raw_text = get_ocr_text(final_path)
        
        prompt = f"""
        You are a Medical Report Interpretation AI. You will analyze ANY type of medical document (blood test, scan report, radiology, ECG, discharge summary, PCR, microbiology, biopsy, or any hospital report) and produce two sections:

SECTION 1: PATIENT EXPLANATION
SECTION 2: DOCTOR SUMMARY

Follow these rules exactly:

GENERAL RULES:

NO markdown. Pure text only.
Do not add facts not found in the report.
Do not guess diagnoses, severities, or abnormal values.
Expand medical abbreviations when needed.
Keep the tone calm, neutral, and medically safe.
If reference ranges are missing, explicitly say so.
If meaning is unclear, say “cannot be determined from this report.”
Never dump a long list of numbers without grouping.
Always separate raw values (findings) and value changes (trends).

SECTION 1: PATIENT EXPLANATION (simple, friendly, non-technical)

Write in short paragraphs. Include:
What kind of test/report this is.
What this type of test usually checks for.
A simple explanation of the important values and trends in the report.
Only interpret what is clearly supported by the given data.
Mention when results cannot be judged because reference ranges or clinical context are missing.
End with: “Only your doctor can confirm what these results mean for you.
The explanation must be easy enough for a person with no medical background.
Use everyday language and avoid medical terms unless necessary.
Keep sentences short and direct.
Never repeat all numbers unless they are essential to explain a trend.

SECTION 2: DOCTOR SUMMARY (fast-reading, point-based clinical notes)

Write in short numbered points, like a clinician's quick-review summary.
Include:
Report type (e.g., biochemistry, CBC, radiology, ECG).
Key findings explicitly mentioned in the report.
Trends or comparisons if multiple samples exist.
Relevant systems involved (hepatic, renal, hematologic, etc.).
Possible meaning of trends ONLY based on data (e.g., “trend suggests resolving leukocytosis”).
Limitations such as missing reference ranges, missing history, missing timestamps.
Any recommendations given in the report (if present).
Clinical follow-up required (e.g., “clinical correlation needed”).
Keep the doctor section concise, high-signal, and strictly based on the provided data.
Group raw values by system (Hepatic, Renal, Electrolytes, Hematology).
Show changes using “X→Y” format instead of listing both numbers separately.
Do not mix Findings and Trends. Findings = raw values. Trends = changes only.
Keep the doctor summary highly structured and compressed for speed-reading.
        REPORT: {raw_text}
        """
        response = llm.invoke(prompt)
        analysis_text = response.content
        
        # Suggest hospitals based on analysis text (using RAG/CSV engine now)
        # Note: We just send the analysis text to the frontend, allowing user to click "Find Hospitals" there
        
        pdf_filename = f"Report_{int(time.time())}.pdf"
        generate_pdf(analysis_text, pdf_filename)

        return jsonify({
            "success": True,
            "analysis": analysis_text,
            "pdf_url": f"/download/{pdf_filename}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/update_map_route', methods=['POST'])
def update_map_route():
    """
    Generates a new map with a route to a SPECIFIC hospital selected by the user.
    """
    data = request.json
    user_coords = data.get('user_coords') # [lat, lon]
    target_hospital = data.get('target_hospital') # {lat, lon, name...}
    all_hospitals = data.get('all_hospitals') # List of all hospitals to keep markers

    if not user_coords or not target_hospital:
        return jsonify({"success": False, "error": "Missing data"})

    # Calculate Route to the SELECTED hospital
    route_geom, _, _ = geo_engine.get_route_geometry(
        user_coords[0], user_coords[1], 
        target_hospital['lat'], target_hospital['lon']
    )

    # Generate new Map HTML
    map_html = geo_engine.generate_map_html_string(user_coords, route_geom, all_hospitals)

    return jsonify({
        "success": True,
        "map_html": map_html
    })

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=True)

@app.route('/assets/<path:filename>')
def assets_file(filename):
    # Serve local sign videos (and any other assets placed in /assets).
    return send_from_directory(ASSETS_FOLDER, filename)

@app.route('/signs-map', methods=['GET'])
def get_signs_map():
    """
    Returns available sign videos as a word->URL map.

    Sources (in priority order):
    1) Dataset folders: assets/dataset/SL/<word>/<video>.(mp4|mov|webm)
       - Maps <word> -> first video file in that folder (sorted).
    2) Flat folder fallback: assets/signs/*.mp4
       - Maps <basename> -> that file.
    """
    mapping = {}

    # 1) Dataset folder structure
    dataset_root = os.path.join(ASSETS_FOLDER, "dataset", "SL")
    try:
        if os.path.isdir(dataset_root):
            for word in os.listdir(dataset_root):
                word_dir = os.path.join(dataset_root, word)
                if not os.path.isdir(word_dir):
                    continue
                files = sorted(os.listdir(word_dir))
                first_video = None
                for fname in files:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in [".mp4", ".mov", ".webm"]:
                        first_video = fname
                        break
                if not first_video:
                    continue
                key = word.strip().lower()
                mapping[key] = f"/assets/dataset/SL/{word}/{first_video}"
    except Exception:
        pass

    # 2) Flat fallback signs folder
    signs_dir = os.path.join(ASSETS_FOLDER, "signs")
    try:
        if os.path.isdir(signs_dir):
            for name in os.listdir(signs_dir):
                if not name.lower().endswith(".mp4"):
                    continue
                key = os.path.splitext(name)[0].strip().lower()
                mapping.setdefault(key, f"/assets/signs/{name}")
    except Exception:
        pass

    # Small alias set for common typos/variants.
    if "consious" in mapping and "conscious" not in mapping:
        mapping["conscious"] = mapping["consious"]

    return jsonify(mapping)

def _append_emergency_log(event: dict) -> str:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    log_path = os.path.join(OUTPUT_FOLDER, "emergency_alerts.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return log_path

def _fetch_firebase_emergency_contacts(patient_id: str) -> list:
    """
    Returns up to 5 emergency contact numbers.
    Priority: EMERGENCY_CONTACTS env var → Firestore REST API → TWILIO_TO_NUMBER fallback.
    """
    # 1. Use the env-var list if populated (most reliable, no network call needed)
    if EMERGENCY_CONTACTS_ENV:
        return EMERGENCY_CONTACTS_ENV[:5]

    # 2. Try Firebase Firestore REST API
    if patient_id and FIREBASE_PROJECT_ID and not FIREBASE_PROJECT_ID.startswith("1:"):
        try:
            url = (
                f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}"
                f"/databases/(default)/documents/emergency_contacts/{patient_id}"
                f"?key={FIREBASE_API_KEY}"
            )
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                fields = resp.json().get("fields", {})
                numbers = [
                    fields.get(f"phone{i}", {}).get("stringValue", "").strip()
                    for i in range(1, 6)
                ]
                numbers = [n for n in numbers if n]
                if numbers:
                    return numbers
        except Exception as e:
            print(f"Firebase contact fetch error: {e}")

    # 3. Last resort: single TWILIO_TO_NUMBER
    return [TWILIO_TO_NUMBER] if TWILIO_TO_NUMBER else []


def _try_send_twilio_alert(message: str, to_numbers: list = None):
    """
    Sends SMS via Twilio to each number in to_numbers (up to 5).
    Falls back to TWILIO_TO_NUMBER if list is empty.
    """
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER):
        return {"sent": False, "reason": "Twilio not configured"}
    if not to_numbers:
        to_numbers = [TWILIO_TO_NUMBER] if TWILIO_TO_NUMBER else []
    if not to_numbers:
        return {"sent": False, "reason": "No recipient numbers"}
    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    except Exception as e:
        return {"sent": False, "reason": f"Twilio client init failed: {e}"}

    results = []
    for number in to_numbers[:5]:
        try:
            sms = client.messages.create(body=message, from_=TWILIO_FROM_NUMBER, to=number)
            results.append({"to": number, "sent": True, "sid": getattr(sms, "sid", None)})
            print(f"[ALERT] SMS sent to {number} — SID: {getattr(sms, 'sid', None)}")
        except Exception as e:
            results.append({"to": number, "sent": False, "error": str(e)})
            print(f"[ALERT] SMS FAILED to {number} — {e}")

    any_sent = any(r["sent"] for r in results)
    return {"sent": any_sent, "results": results}

@app.route('/predict_sign', methods=['POST'])
def predict_sign():
    """
    Backend model-serving hook for patient webcam sign inference.
    Browser sends a pretrained MediaPipe gesture label + confidence.
    We map it to a medically useful token and apply low-confidence fallback.
    """
    payload = request.get_json(silent=True) or {}
    gesture = str(payload.get("gesture", "")).strip()
    confidence = float(payload.get("confidence", 0.0) or 0.0)

    # Safety fallback for uncertain frames
    if confidence < 0.70 or not gesture:
        return jsonify({
            "success": True,
            "word": "unknown",
            "confidence": confidence,
            "gesture": gesture,
        })

    # Pretrained MediaPipe gesture label -> token map (expanded)
    gesture_to_word = {
        # Existing mappings
        "Open_Palm": "help",
        "Closed_Fist": "pain",
        "Pointing_Up": "where",
        "Victory": "ok",
        "Thumb_Up": "yes",
        "Thumb_Down": "no",
        "ILoveYou": "you",
        "Fist": "emergency",
        "Peace": "calm",
        "Call_Me": "call",
        "Stop_Sign": "stop",
        "Live_Long": "breathe",
        "Rock": "music",
        "One": "one",
        "Two": "two",
        "Three": "three",
        "Four": "four",
        "Five": "five",
        # Expanded medical/health terms
        "Pain": "pain",
        "Chest": "chest",
        "Head": "head",
        "Headache": "headache",
        "Stomach": "stomach",
        "Throat": "throat",
        "Fever": "fever",
        "Cold": "cold",
        "Cough": "cough",
        "Vomiting": "vomiting",
        "Diarrhea": "diarrhea",
        "Dizziness": "dizziness",
        "Dizzy": "dizzy",
        "Breath": "breath",
        "Breathe": "breathe",
        "Weakness": "weakness",
        "Weak": "weak",
        "Tired": "tired",
        "Fatigue": "fatigue",
        "Swelling": "swelling",
        "Injury": "injury",
        "Bleeding": "bleeding",
        "Help": "help",
        "Emergency": "emergency",
        "Ambulance": "ambulance",
        "Urgent": "urgent",
        "Severe": "severe",
        "Critical": "critical",
        "Unconscious": "unconscious",
        "Faint": "faint",
        "Collapse": "collapse",
        "Attack": "attack",
        "Heart": "heart",
        "Lungs": "lungs",
        "Brain": "brain",
        "Eye": "eye",
        "Ear": "ear",
        "Nose": "nose",
        "Mouth": "mouth",
        "Neck": "neck",
        "Back": "back",
        "Arm": "arm",
        "Leg": "leg",
        "Hand": "hand",
        "Foot": "foot",
        "Skin": "skin",
        "Medicine": "medicine",
        "Tablet": "tablet",
        "Injection": "injection",
        "Treatment": "treatment",
        "Surgery": "surgery",
        "Test": "test",
        "Scan": "scan",
        "Report": "report",
        "Diagnosis": "diagnosis",
        "Doctor": "doctor",
        "Nurse": "nurse",
        "Hospital": "hospital",
        "Clinic": "clinic",
        "Pharmacy": "pharmacy",
        "Prescription": "prescription",
        "Dose": "dose",
        "Blood": "blood",
        "Pressure": "pressure",
        "Sugar": "sugar",
        "Oxygen": "oxygen",
        # Existing and other terms
        "Bandage": "injury",
        "Pill": "medicine",
        "Stethoscope": "doctor",
        "Breath": "breath",
        "Tooth": "tooth",
        "Diabetes": "diabetes",
        "Asthma": "asthma",
        "Cancer": "cancer",
        "Infection": "infection",
        "Allergy": "allergy",
        "Hungry": "hungry",
        "Thirsty": "thirsty",
        "Sleep": "sleep",
        "Confused": "confused",
        "Burn": "burn",
        "Rash": "rash",
        "Fracture": "fracture",
        "Paralysis": "paralysis",
        "Seizure": "seizure",
        "Stroke": "stroke",
        "Heart_Attack": "heart attack",
        "Poison": "poison",
        "Choking": "choking",
        "Constipation": "constipation",
        "Urine": "urine",
        "Pregnant": "pregnant",
        "Baby": "baby",
        "Old": "elderly",
        "Child": "child",
        "Male": "male",
        "Female": "female",
    }

    word = gesture_to_word.get(gesture, "unknown")
    return jsonify({
        "success": True,
        "word": word,
        "confidence": confidence,
        "gesture": gesture,
    })

@app.route('/alert', methods=['POST'])
def alert():
    """
    Receives emergency alerts from the patient-side emotion monitor.
    Expected JSON:
      - condition: str
      - confidence: float (0-1)
      - timestamp_ms: int
      - location: {lat, lon, accuracy_m} | null
      - room: str | null
      - patient_id: str | null  (used to fetch Firebase emergency contacts)
    """
    payload = request.get_json(silent=True) or {}
    condition  = str(payload.get("condition", "")).strip()
    confidence = payload.get("confidence")
    timestamp_ms = payload.get("timestamp_ms")
    location   = payload.get("location")
    room       = payload.get("room")
    patient_id = str(payload.get("patient_id", "") or "").strip()

    if not condition:
        return jsonify({"success": False, "error": "Missing condition"}), 400

    event = {
        "type": "emergency_alert",
        "received_at_ms": int(time.time() * 1000),
        "condition": condition,
        "confidence": confidence,
        "timestamp_ms": timestamp_ms,
        "location": location,
        "room": room,
        "patient_id": patient_id,
        "user_agent": request.headers.get("User-Agent", ""),
        "remote_addr": request.remote_addr,
    }

    log_path = _append_emergency_log(event)

    lat = lon = None
    if isinstance(location, dict):
        lat = location.get("lat")
        lon = location.get("lon")

    if lat is not None and lon is not None:
        maps_link = f"https://maps.google.com/?q={lat},{lon}"
    else:
        maps_link = None

    msg = (
        f"\U0001f6a8 EMERGENCY ALERT \U0001f6a8\n"
        f"Patient may be unconscious.\n"
    )
    if maps_link:
        msg += f"Location: {maps_link}\n"
    msg += "Please respond immediately."

    contacts = _fetch_firebase_emergency_contacts(patient_id)
    print(f"[ALERT] condition={condition} patient_id={patient_id!r} location={location} contacts={contacts}")
    twilio_result = _try_send_twilio_alert(msg, contacts)
    print(f"[ALERT] Twilio result: {twilio_result}")
    event["twilio"] = twilio_result
    _append_emergency_log({"type": "emergency_alert_delivery", "event": event})

    # Emit nearest hospitals to doctor in real-time via Socket.IO
    hospitals_payload = []
    print(f"[ALERT] Hospital lookup — lat={lat} lon={lon} room={room!r}")

    if lat is not None and lon is not None:
        try:
            raw = hospital_engine.find_nearest_n(lat, lon, n=5)[:5]
            for h in raw:
                hospitals_payload.append({
                    "name":    h.get("name", "Unknown"),
                    "address": h.get("address", ""),
                    "dist":    h.get("dist", 0),
                    "time":    h.get("time", "N/A"),
                    "lat":     h.get("lat"),
                    "lon":     h.get("lon"),
                    "phone":   "108",
                })
            print(f"[ALERT] Found {len(hospitals_payload)} hospitals: {[h['name'] for h in hospitals_payload]}")
        except Exception as e:
            print(f"[ALERT] Hospital lookup failed: {e}")
    else:
        print("[ALERT] No location — skipping hospital distance lookup")

    # Resolve doctor's socket SID directly — avoids room name mismatch entirely
    doctor_sid = None
    if room and room in active_rooms:
        doctor_sid = active_rooms[room].get("doctor_sid")

    payload = {
        "condition":  condition,
        "patient_id": patient_id,
        "location":   location,
        "hospitals":  hospitals_payload,
    }

    if doctor_sid:
        print(f"[ALERT] Emitting nearest_hospitals directly to doctor SID={doctor_sid} (room={room!r}, hospitals={len(hospitals_payload)})")
        socketio.emit("nearest_hospitals", payload, to=doctor_sid)
    elif room:
        print(f"[ALERT] No doctor SID found — falling back to room emit room={room!r}")
        socketio.emit("nearest_hospitals", payload, room=room)
    else:
        print("[ALERT] No room or SID — broadcasting nearest_hospitals to all clients")
        socketio.emit("nearest_hospitals", payload)

    return jsonify({
        "success": True,
        "logged_to": log_path,
        "twilio": twilio_result,
        "contacts_notified": len(contacts),
        "hospitals_found": len(hospitals_payload),
    })

def _find_nearest_hospitals_fast(lat, lon, n=5):
    """
    Pure-Haversine hospital lookup — no OSRM, no blocking network call.
    ETA = (distance_km / 40) * 60  (assumes 40 km/h ambulance speed).
    Returns list of dicts: name, address, dist, time, phone.
    """
    if hospital_engine.data is None or lat is None or lon is None:
        return []
    df = hospital_engine.data.copy()
    lc  = hospital_engine.lat_col
    lnc = hospital_engine.lon_col
    nc  = hospital_engine.name_col
    ac  = hospital_engine.addr_col

    df["_dist_km"] = df.apply(
        lambda r: haversine(lat, lon, r[lc], r[lnc]), axis=1
    )
    nearest = df.nsmallest(n, "_dist_km")

    results = []
    for _, row in nearest.iterrows():
        dist = round(float(row["_dist_km"]), 2)
        eta  = max(1, round((dist / 40) * 60))
        results.append({
            "name":    str(row[nc]),
            "address": str(row[ac])[:50],
            "dist":    dist,
            "time":    f"{eta} mins",
            "phone":   "108",
        })
    return results


@socketio.on("patient_location")
def handle_patient_location(data):
    """
    Patient emits: { lat, lng, room, condition, patient_id }
    Server finds 5 nearest hospitals via Haversine and emits
    'nearest_hospitals' directly to the doctor in the same room.
    """
    room       = data.get("room", "")
    lat        = data.get("lat")
    lng        = data.get("lng") or data.get("lon")
    condition  = str(data.get("condition", "unconscious"))
    patient_id = str(data.get("patient_id", ""))

    print(f"[patient_location] room={room!r} lat={lat} lng={lng} condition={condition}")

    hospitals = _find_nearest_hospitals_fast(lat, lng, n=5)
    print(f"[patient_location] found {len(hospitals)} hospitals")

    payload = {
        "condition":  condition,
        "patient_id": patient_id,
        "location":   {"lat": lat, "lon": lng} if lat and lng else None,
        "hospitals":  hospitals,
    }

    # Prefer direct SID delivery to avoid room-name mismatch
    doctor_sid = active_rooms.get(room, {}).get("doctor_sid") if room else None
    if doctor_sid:
        print(f"[patient_location] emitting nearest_hospitals → doctor SID={doctor_sid}")
        emit("nearest_hospitals", payload, to=doctor_sid)
    elif room:
        print(f"[patient_location] no doctor SID — emitting to room={room!r}")
        emit("nearest_hospitals", payload, to=room)
    else:
        print("[patient_location] no room — broadcasting nearest_hospitals")
        emit("nearest_hospitals", payload, broadcast=True)


@app.route('/debug_emit')
def debug_emit():
    """
    Test route: manually fires nearest_hospitals to every connected socket.
    Usage: open http://localhost:10000/debug_emit in browser while doctor tab is open.
    """
    room = request.args.get("room", "")
    doctor_sid = None
    if room and room in active_rooms:
        doctor_sid = active_rooms[room].get("doctor_sid")

    dummy_payload = {
        "condition":  "unconscious",
        "patient_id": "debug-test",
        "location":   {"lat": 13.0827, "lon": 80.2707},
        "hospitals": [
            {"name": "Apollo Hospital (Test)",   "address": "Greams Lane, Chennai", "dist": 1.2, "time": "4 mins",  "phone": "108"},
            {"name": "MIOT International (Test)", "address": "Manapakkam, Chennai",  "dist": 3.8, "time": "11 mins", "phone": "108"},
            {"name": "Fortis Malar (Test)",       "address": "Adyar, Chennai",        "dist": 5.1, "time": "15 mins", "phone": "108"},
            {"name": "Billroth Hospital (Test)",  "address": "Shenoy Nagar, Chennai", "dist": 6.4, "time": "19 mins", "phone": "108"},
            {"name": "Sri Ramachandra (Test)",    "address": "Porur, Chennai",        "dist": 9.0, "time": "25 mins", "phone": "108"},
        ],
    }

    if doctor_sid:
        socketio.emit("nearest_hospitals", dummy_payload, to=doctor_sid)
        msg = f"Emitted to doctor SID={doctor_sid} (room={room!r})"
    elif room:
        socketio.emit("nearest_hospitals", dummy_payload, room=room)
        msg = f"Emitted to room={room!r} (no doctor SID found — used room fallback)"
    else:
        socketio.emit("nearest_hospitals", dummy_payload)
        msg = "Broadcast to ALL connected sockets (no room specified)"

    print(f"[DEBUG_EMIT] {msg}")
    print(f"[DEBUG_EMIT] active_rooms={dict(active_rooms)}")
    return f"<pre>OK — {msg}\n\nactive_rooms={dict(active_rooms)}</pre>"

# --- VIDEO & AGORA ROUTES (MERGED) ---

@app.route('/get-token', methods=['GET'])
def get_token():
    if not AZURE_SPEECH_KEY:
        return jsonify({
            "error": "Missing AZURE_SPEECH_KEY. Add it to .env (and restart the server)."
        }), 500
    fetch_token_url = f"https://{AZURE_REGION}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
    headers = {'Ocp-Apim-Subscription-Key': AZURE_SPEECH_KEY}
    try:
        response = requests.post(fetch_token_url, headers=headers)
        response.raise_for_status()
        return jsonify({"token": response.text, "region": AZURE_REGION})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-glossary', methods=['GET'])
def get_glossary():
    return jsonify(load_glossary())

# --- SOCKET EVENTS ---

@socketio.on('join_request')
def handle_join_request(data):
    room = data['room']
    role = data['role'] 
    sid = request.sid
    
    # Generate simple integer UID for Agora
    agora_uid = int(str(abs(hash(sid)))[:8])

    if room not in active_rooms:
        active_rooms[room] = {'doctor_sid': None, 'patients': []}

    # --- 1. DOCTOR JOIN LOGIC ---
    if role == 'doctor':
        current_doc = active_rooms[room]['doctor_sid']
        
        # Strict Check: If doctor exists and it's not me reconnecting -> Block
        if current_doc is not None and current_doc != sid:
            emit('join_response', {'allowed': False, 'message': '⛔ ACCESS DENIED: Another Doctor is already in this session.'})
            return 
        
        active_rooms[room]['doctor_sid'] = sid
        join_room(room)
        
        token = generate_agora_token(room, agora_uid)
        
        emit('join_response', {
            'allowed': True, 
            'role': 'doctor',
            'agora_app_id': AGORA_APP_ID,
            'agora_token': token,
            'agora_uid': agora_uid
        })
        broadcast_room_status(room)
        print(f"✅ Doctor joined {room}")

    # --- 2. PATIENT JOIN LOGIC ---
    elif role == 'patient':
        # Strict Check: Max 1 Patient
        if len(active_rooms[room]['patients']) > 0 and sid not in active_rooms[room]['patients']:
            emit('join_response', {'allowed': False, 'message': '⛔ ACCESS DENIED: The session is full (Patient already present).'})
            return 
        
        if sid not in active_rooms[room]['patients']:
            active_rooms[room]['patients'].append(sid)
            
        join_room(room)
        
        token = generate_agora_token(room, agora_uid)
        
        emit('join_response', {
            'allowed': True, 
            'role': 'patient',
            'agora_app_id': AGORA_APP_ID,
            'agora_token': token,
            'agora_uid': agora_uid
        })
        broadcast_room_status(room)
        print(f"✅ Patient joined {room}")

@socketio.on('send_translation')
def handle_translation(data):
    room = data['room']
    emit('receive_translation', data, room=room, include_self=False)

@socketio.on('send_sign_sentence')
def handle_sign_sentence(data):
    """
    Doctor -> Patient: send a normalized word list for sign-language playback.
    """
    room = data.get('room')
    if not room:
        return
    emit('receive_sign_sentence', data, room=room, include_self=False)

@socketio.on('send_patient_sign_text')
def handle_patient_sign_text(data):
    """
    Patient -> Doctor live recognized sign text.
    """
    room = data.get('room')
    if not room:
        return
    emit('receive_patient_sign_text', data, room=room, include_self=False)

@socketio.on('end_meeting')
def handle_end_meeting(data):
    room = data.get('room')
    sid = request.sid
    # Only allow if the requester is the Doctor of that room
    if room in active_rooms and active_rooms[room]['doctor_sid'] == sid:
        shutdown_room(room, "The Doctor has ended the consultation.")

@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    for room, info in list(active_rooms.items()):
        # If Doctor leaves -> End for all
        if info['doctor_sid'] == sid:
            shutdown_room(room, "The Doctor has disconnected.")
            return
        # If Patient leaves -> Just remove them
        if sid in info['patients']:
            info['patients'].remove(sid)
            broadcast_room_status(room)

@socketio.on('leave')
def on_leave(data):
    leave_room(data['room'])
    on_disconnect()

def shutdown_room(room, message):
    if room in active_rooms:
        emit('meeting_ended', {'message': message}, room=room)
        close_room(room)
        del active_rooms[room]
        print(f"🛑 Room {room} destroyed.")

@app.route('/prescription')
def prescription():
    return render_template('prescription.html')

@app.route('/send_prescription', methods=['POST'])
def send_prescription():
    try:
        data = request.get_json()
        room = data.get('room', '')

        payload = {
            'patient_name': data.get('patientName', ''),
            'doctor_name':  data.get('doctorName', ''),
            'date':         data.get('prescriptionDate', ''),
            'medicines':    data.get('medicineName', []),
            'notes':        data.get('notes', '')
        }

        if room:
            socketio.emit('receive_prescription', payload, room=room)

        return jsonify({'success': True})

    except Exception as e:
        print(f"Error sending prescription: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/generate_prescription', methods=['POST'])
def generate_prescription():
    try:
        data = request.get_json()
        
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Set font
        pdf.set_font('Arial', 'B', 16)
        
        # Header
        pdf.cell(0, 10, 'MEDICAL PRESCRIPTION', 0, 1, 'C')
        pdf.ln(10)
        
        # Prescription info
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 8, f'Date: {data.get("prescriptionDate", "")}', 0, 1)
        pdf.cell(0, 8, f'Doctor: {data.get("doctorName", "")}', 0, 1)
        pdf.ln(5)
        
        # Patient info
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Patient Information:', 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 8, f'Name: {data.get("patientName", "")}', 0, 1)
        
        if data.get("patientAge"):
            pdf.cell(0, 8, f'Age: {data.get("patientAge", "")}', 0, 1)
        if data.get("patientGender"):
            pdf.cell(0, 8, f'Gender: {data.get("patientGender", "")}', 0, 1)
        if data.get("patientContact"):
            pdf.cell(0, 8, f'Contact: {data.get("patientContact", "")}', 0, 1)
        
        pdf.ln(5)
        
        # Diagnosis
        if data.get("diagnosis"):
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, 'Diagnosis:', 0, 1)
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 8, data.get("diagnosis", ""), 0, 1)
            pdf.ln(5)
        
        # Medicines
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Medicines:', 0, 1)
        pdf.set_font('Arial', '', 12)
        
        medicine_names = data.get('medicineName', [])
        medicine_dosages = data.get('medicineDosage', [])
        medicine_frequencies = data.get('medicineFrequency', [])
        
        for i, name in enumerate(medicine_names):
            if name.strip():
                dosage = medicine_dosages[i] if i < len(medicine_dosages) else ''
                frequency = medicine_frequencies[i] if i < len(medicine_frequencies) else ''
                
                pdf.cell(0, 8, f'{i+1}. {name}', 0, 1)
                if dosage:
                    pdf.cell(10, 8, '', 0, 0)  # Indent
                    pdf.cell(0, 8, f'Dosage: {dosage}', 0, 1)
                if frequency:
                    pdf.cell(10, 8, '', 0, 0)  # Indent
                    pdf.cell(0, 8, f'Frequency: {frequency}', 0, 1)
                pdf.ln(2)
        
        # Notes
        if data.get("notes"):
            pdf.ln(5)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, 'Additional Notes/Instructions:', 0, 1)
            pdf.set_font('Arial', '', 12)
            # Handle multi-line notes
            notes = data.get("notes", "")
            lines = pdf.multi_cell(0, 8, notes)
            pdf.ln(5)
        
        # Footer
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 8, 'Generated by Tele-Health AI Platform', 0, 1, 'C')
        
        # Save PDF to memory
        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        pdf_output.seek(0)
        
        # Create response
        response = make_response(pdf_output.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=prescription_{data.get("patientName", "unknown")}_{data.get("prescriptionDate", "unknown")}.pdf'
        
        return response
        
    except Exception as e:
        print(f"Error generating prescription: {str(e)}")
        return jsonify({"error": "Failed to generate prescription"}), 500

@app.route('/test')
def test():
    return "Server is working ✅"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    print("🚀 Attempting to start server...") # Added print to see progress
    # CRITICAL CHANGE: debug=False prevents the reloader from hanging with TensorFlow
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)