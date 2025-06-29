import sys
from datetime import datetime

sys.path.insert(1, r'/demo_working/firebase_auth.py')
import firebase_admin
from firebase_admin import credentials, db

def initialize_firebase():
    """Initialize Firebase and return a database reference"""
    cred = credentials.Certificate(r"E:\UsualDev_Hackhaven\traffic-monitoring-f490d-firebase-adminsdk-fbsvc-bd36dd8ff2.json")  # Update with your correct path
    if not firebase_admin._apps:  # Prevents re-initialization error
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://traffic-monitoring-f490d-default-rtdb.firebaseio.com/'  # Replace with your Firebase URL
        })
    return db.reference("traffic_data")  # Returns database reference


# initialize_firebase()

import numpy as np

# Transformation matrix from camera calibration
H = np.array([
    [-0.027557, -0.050103, 42.587],
    [-2.9606e-17, -0.52608, 215.69],
    [0, -0.0031216, 1]
])

def pixel_to_gps(cx, cy):
    pixel_coords = np.array([cx, cy, 1])  # Convert to homogeneous coordinates
    world_coords = np.dot(H, pixel_coords)  # Apply transformation
    world_coords /= world_coords[2]  # Normalize

    # Assuming reference point (latitude_ref, longitude_ref) and scale
    latitude_ref = 28.634556# Reference latitude (Example: New Delhi)
    longitude_ref = 77.448029  # Reference longitude

    scale_factor = 0.00001  # Approximate conversion factor (adjust as needed)
    lat = latitude_ref + (world_coords[1] * scale_factor)
    lon = longitude_ref + (world_coords[0] * scale_factor)

    return lat, lon





def crash_to_fire(  cx,cy,track_id):
    ref=initialize_firebase()

    lat, lon = pixel_to_gps(cx, cy)  # Convert pixel to GPS
    timestamp = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    # track_id = int(float(track_id))  # Convert track_id to an integer

    accident_ref = ref.child("crash_vehicle").child(str(track_id))
    accident_ref.set({

        "latitude": str(lat),
        "vehicle_id": track_id,
        "longitude": str(lon),
        "timestamp": timestamp
    })




crash_to_fire(205 , 75 , 1)