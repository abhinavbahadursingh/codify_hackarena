import csv
from collections import defaultdict
from firebase_auth import initialize_firebase  # Import Firebase authentication
import pandas as pd
import time
from datetime import datetime
# Get database reference
ref = initialize_firebase()

# Set to store already tracked vehicles
tracked_vehicles = set()

csv_file = "traffic_speed_data.csv"

# Function to store speed in CSV
def store_speed(track_id, speed):
    """Stores the track_id, timestamp, and speed in a CSV file."""

    # Create a DataFrame for the new entry
    new_entry = pd.DataFrame({
        "track_id": [track_id],
        "timestamp": [datetime.now().strftime("%H:%M:%S")],  # Current time as timestamp
        "speed": [speed]
    })

    # Append to CSV file (create if not exists)
    try:
        new_entry.to_csv(csv_file, mode='a', header=not pd.io.common.file_exists(csv_file), index=False)
    except Exception as e:
        print(f"Error writing to CSV: {e}")


# Function to calculate average speed in 5-second intervals
def calculate_average_speed():
    """Calculates average speed for each track_id in 5-second intervals."""

    try:
        df = pd.read_csv(csv_file)  # Read CSV
    except FileNotFoundError:
        print("CSV file not found!")
        return

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%H:%M:%S")

    # Sort by track_id and timestamp
    df.sort_values(by=["track_id", "timestamp"], inplace=True)

    # Resample data in 5-second intervals and calculate the mean speed
    df["interval"] = df["timestamp"].dt.floor("5S")  # Round timestamps to nearest 5s
    avg_speed = df.groupby(["track_id", "interval"])["speed"].mean().reset_index()

    print("\nAverage Speed per 5-second Interval:")
    print(avg_speed)


def log_speed(track_id, speed):
    track_id = int(float(track_id))  # Convert track_id to an integer

    speed_ref = ref.child("speed_Data").child("real_Time_Speed").child(str(track_id))
    store_speed(track_id,speed)
    # Get the current count from Firebase
    count_ref = speed_ref.child("count")
    count_snapshot = count_ref.get()

    if count_snapshot is None:
        count = 1  # Initialize count if it doesn't exist
    else:
        count = count_snapshot + 1  # Increment count

    # Store speed with unique key
    speed_ref.child(f"time{count}").set({"speed": speed})

    # Update the count for the next entry
    count_ref.set(count)
