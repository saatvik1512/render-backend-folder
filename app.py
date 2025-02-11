import os
import uuid
import cv2
import numpy as np
import face_recognition
from flask import Flask, request, jsonify
from flask_cors import CORS
import pyttsx3
from ultralytics import YOLO
from google.generativeai import GenerativeModel, configure

configure(api_key=os.getenv('GOOGLE_API_KEY'))  # Set in .env
gemini = GenerativeModel('gemini-pro')

app = Flask(__name__)
CORS(app)

engine = pyttsx3.init()
model = YOLO('yolov8n.pt')  
# Configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, 'known_faces')
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

def save_face_encoding(user_id, image_path):
    """Save face encoding to file"""
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    
    if len(encodings) == 0:
        return False
        
    np.save(os.path.join(KNOWN_FACES_DIR, f"{user_id}.npy"), encodings[0])
    return True

def recognize_face(image_path):
    """Compare face with known encodings"""
    unknown_image = face_recognition.load_image_file(image_path)
    unknown_encoding = face_recognition.face_encodings(unknown_image)
    
    if not unknown_encoding:
        return None
        
    unknown_encoding = unknown_encoding[0]
    
    for file_name in os.listdir(KNOWN_FACES_DIR):
        if file_name.endswith(".npy"):
            user_id = os.path.splitext(file_name)[0]
            known_encoding = np.load(os.path.join(KNOWN_FACES_DIR, file_name))
            
            results = face_recognition.compare_faces([known_encoding], unknown_encoding)
            if results[0]:
                return user_id
                
    return None

@app.route('/get-object-info', methods=['POST'])
def get_object_info():
    data = request.json
    if not data.get('object'):
        print("No object specified")
        return jsonify({"error": "No object specified"}), 400
    
    try:
        response = gemini.generate_content(
            f"Give a concise 50-word description of {data['object']} in simple English. Focus on key characteristics and common uses."
        )
        return jsonify({"info": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/detect-objects', methods=['POST'])
def detect_objects():
    if 'file' not in request.files:
        print("No file uploaded")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    temp_path = os.path.join(TEMP_DIR, f"obj_{uuid.uuid4()}.jpg")
    
    try:
        # Save and process image
        file.save(temp_path)
        img = cv2.imread(temp_path)
        
        # YOLO object detection
        results = model(img)
        boxes = results[0].boxes.xyxy.tolist()
        classes = results[0].boxes.cls.tolist()
        names = results[0].names
        
        # Dimension calculation using reference object (A4 paper)
        ref_width_cm = 21.0  # A4 paper width
        ref_objects = [i for i, cls in enumerate(classes) if names[int(cls)] == 'book']
        if not ref_objects:
            print("Place a reference object (book) in frame")
            return jsonify({"error": "Place a reference object (book) in frame"}), 400
            
        ref_box = boxes[ref_objects[0]]
        ref_width_px = ref_box[2] - ref_box[0]
        px_per_cm = ref_width_px / ref_width_cm
        
        # Process detections
        detections = []
        for box, cls in zip(boxes, classes):
            label = names[int(cls)]
            x1, y1, x2, y2 = box
            width = round((x2 - x1) / px_per_cm, 1)
            height = round((y2 - y1) / px_per_cm, 1)
            
            detections.append({
                "label": label,
                "width": width,
                "height": height,
                "bbox": [x1, y1, x2, y2]
            })
        
        return jsonify({"results": detections})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/signup', methods=['POST'])
def signup():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    try:
        file = request.files['file']
        in_memory_file = file.read()
        nparr = np.frombuffer(in_memory_file, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            print("Erioriririri")
            return jsonify({'error': 'Invalid image format'}), 400
        

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        temp_path = os.path.join(TEMP_DIR, f"temp_{uuid.uuid4()}.jpg")
        cv2.imwrite(temp_path, rgb_img)

    
    # Check if face is detected
        image = face_recognition.load_image_file(temp_path)
        face_locations = face_recognition.face_locations(image)
        
        if len(face_locations) == 0:
            os.remove(temp_path)
            return jsonify({'error': 'No face detected'}), 400
            
        if len(face_locations) > 1:
            os.remove(temp_path)
            return jsonify({'error': 'Multiple faces detected'}), 400
        
        existing_user = recognize_face(temp_path)
        if existing_user is not None:
            os.remove(temp_path)
            engine.say("User already exists. Please login.")
            engine.runAndWait()
            return jsonify({'error': 'User already exists. Please login.'}), 400

        # Create new user
        user_id = str(uuid.uuid4())
        if not save_face_encoding(user_id, temp_path):
            os.remove(temp_path)
            return jsonify({'error': 'Face encoding failed'}), 500
            
        os.remove(temp_path)
        return jsonify({'message': 'Registration successful', 'userId': user_id}), 200
    except Exception as e:
        return jsonify({'error': f'Image processing failed: {str(e)}'}), 500

@app.route('/signin', methods=['POST'])
def signin():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    temp_path = os.path.join(TEMP_DIR, f"temp_{uuid.uuid4()}.jpg")
    file.save(temp_path)
    
    user_id = recognize_face(temp_path)
    os.remove(temp_path)
    
    if user_id:
        return jsonify({'userId': user_id}), 200
    return jsonify({'error': 'Face not recognized'}), 401

if __name__ == '__main__':
    app.run(debug=True)