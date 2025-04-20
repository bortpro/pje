from djitellopy import Tello
import time
import sys # Import sys to exit gracefully

# E938E1
# Create a Tello object
tello = Tello()

try:
    # --- Connection ---
    print("Connecting to Tello...")
    tello.connect()
    print("Connected!")

    # --- Check Battery ---
    battery_level = tello.get_battery()
    print(f"Battery level: {battery_level}%")
    if battery_level < 20:
        print("Battery low! Please charge the drone.")
        sys.exit("Exiting due to low battery.") # Exit the script

    # --- Flight Plan ---
    print("Taking off...")
    tello.takeoff()
    # Note: Tello SDK commands often have minimum values.
    # move_up minimum is 20cm. We'll use that instead of 10cm.
    print("Moving up 20 cm...")
    tello.move_up(20) # Move up 20 centimeters

    print("Holding position for 5 seconds...")
    time.sleep(10) # Wait for 5 seconds

    print("Landing...")
    tello.land()
    print("Landed safely!")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Attempting emergency landing...")
    try:
        tello.land()
    except Exception as landing_error:
        print(f"Could not execute emergency land: {landing_error}")

finally:
    # Ensure the connection is closed properly, though land() often handles this.
    # For this simple script, just ensuring land() was called is usually enough.
    print("Script finished.")
