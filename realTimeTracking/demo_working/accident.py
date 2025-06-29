
from firebase_auth import initialize_firebase  # Import Firebase authentication
from gps import pixel_to_gps

ref = initialize_firebase()

from datetime import datetime
import time

def pushAccident(track_id, cx, cy):

    lat, lon =pixel_to_gps(cx, cy)  # Convert pixel to GPS
    timestamp = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    track_id = int(float(track_id))  # Convert track_id to an integer

    accident_ref = ref.child("vehicle_Breakdown").child(str(track_id))
    accident_ref.set({

        "latitude": str(lat),
        "vehicle_id": track_id,
        "longitude": str(lon),
        "timestamp": timestamp
    })


