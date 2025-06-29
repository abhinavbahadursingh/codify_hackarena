from firebase_auth import initialize_firebase  # Import Firebase authentication
from gps import pixel_to_gps
from datetime import datetime

# Get database reference
ref = initialize_firebase()

# Set to store already tracked vehicles
tracked_vehicles = set()


def track_vehicle(track_id,cx,cy,name):
    # Send test data
    lat, lon =pixel_to_gps(cx, cy)  # Convert pixel to GPS
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    track_id = int(track_id)
    timestamp =str(timestamp)

    data = {"vehicle_id": track_id, "longitude:":str(lat),"latitude ":str(lon),"entry time ":timestamp}


    vehicle_ref = ref.child("vehicle_Data").child("total_vehicle").child(str(track_id))
    vehicle_ref.set(data)

    type_ref = ref.child("vehicle_Data").child("type_vehicle").child(str(name)).child(str(track_id))
    type_ref.set(data)









