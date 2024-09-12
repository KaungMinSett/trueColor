import cv2
import numpy as np
import mediapipe as mp
from sklearn.cluster import KMeans

def classify_skin_tone(image_file):
    # Read the image
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    
    # Detect faces
    results = face_detection.process(image_rgb)
    
    if not results.detections:
        return "No face detected."
    
    # Use the first detected face
    face = results.detections[0]
    bboxC = face.location_data.relative_bounding_box
    ih, iw, _ = image_rgb.shape
    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
    face_roi = image_rgb[y:y+h, x:x+w]
    
    # Reshape the image to be a list of pixels
    pixels = face_roi.reshape((-1, 3))
    
    # Perform k-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(pixels)
    
    # Get the dominant color (centroid with highest count)
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    
    # Normalize the dominant color
    r, g, b = dominant_color / 255.0
    
    # Calculate color properties
    brightness = (r + g + b) / 3
    chroma = max(r, g, b) - min(r, g, b)
    warmth = r - b
    
    # Determine primary aspect
    if brightness > 0.7:
        primary = "light"
    elif brightness < 0.5:
        primary = "dark"
    else:
        # For medium brightness, consider chroma and warmth
        if chroma > 0.25:
            primary = "bright"
        elif chroma < 0.15:
            primary = "muted"
        elif warmth > 0.05:
            primary = "warm"
        elif warmth < -0.05:
            primary = "cool"
        else:
            primary = "neutral"
        
    # Determine secondary aspect
    if primary not in ["light", "dark"]:
        if warmth > 0.2:
            secondary = "warm"
        elif warmth < 0.15:
            secondary = "cool"
        elif chroma > 0.2:
            secondary = "bright"
        elif chroma < 0.15:
            secondary = "muted"
        else:
            secondary = "neutral"
    else:
        if warmth > 0.2:
            secondary = "warm"
        elif warmth < 0.15:
            secondary = "cool"
        elif chroma > 0.2:
            secondary = "bright"
        else:
            secondary = "muted"
    
    # Map to color season
    color_season = {
        ("light", "warm"): "Light Spring",
        ("light", "cool"): "Light Summer",
        ("bright", "warm"): "Bright Spring",
        ("bright", "cool"): "Bright Winter",
        ("warm", "bright"): "True Spring",
        ("warm", "muted"): "Soft Autumn",
        ("cool", "bright"): "True Winter",
        ("cool", "muted"): "True Summer",
        ("muted", "warm"): "Soft Autumn",
        ("muted", "cool"): "Soft Summer",
        ("dark", "warm"): "Deep Autumn",
        ("dark", "cool"): "Deep Winter",
    }
    
    result = color_season.get((primary, secondary), "Neutral")
    
    # Print debug information
    print(f"Dominant Color (RGB): {dominant_color}")
    print(f"Brightness: {brightness:.2f}, Chroma: {chroma:.2f}, Warmth: {warmth:.2f}")
    print(f"Primary: {primary}, Secondary: {secondary}")
    print(f"Result: {result}")
    
    return result