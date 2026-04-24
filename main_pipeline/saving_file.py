import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==========================================
# Configuration
# ==========================================
SERIAL_PORT = 'COM6'
BAUD_RATE = 9600
FS = 100
WINDOW_SIZE = 500
OUTPUT_FILE = "ecg_data.csv"

# ==========================================
# Serial Port Setup
# ==========================================
print(f"Attempting to connect to {SERIAL_PORT}...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    time.sleep(2)
    print("Connected successfully!")
except Exception as e:
    print(f"Error opening serial port:\n{e}")
    exit()

# ==========================================
# File Setup
# ==========================================
file = open(OUTPUT_FILE, "w")
file.write("timestamp,value\n")  # CSV header

# ==========================================
# Plotting Setup
# ==========================================
fig, ax = plt.subplots(figsize=(10, 4))

raw_data = np.zeros(WINDOW_SIZE)
time_data = np.linspace(0, WINDOW_SIZE / FS, WINDOW_SIZE)

line, = ax.plot(time_data, raw_data)

ax.set_title("Live Raw AD8232 Output")
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Amplitude")
ax.set_ylim(0, 1023)
ax.grid(True)

plt.tight_layout()

# ==========================================
# Animation Loop
# ==========================================
def update(frame):
    global raw_data
    
    while ser.in_waiting:
        try:
            line_in = ser.readline().decode('utf-8').strip()
            if line_in.isdigit():
                val = float(line_in)

                # Save to file
                timestamp = time.time()
                file.write(f"{timestamp},{val}\n")

                # Update plot buffer
                raw_data = np.roll(raw_data, -1)
                raw_data[-1] = val

        except:
            pass

    line.set_ydata(raw_data)
    return line,

ani = FuncAnimation(fig, update, interval=20, blit=False, cache_frame_data=False)

plt.show()

# ==========================================
# Cleanup
# ==========================================
ser.close()
file.close()
print("Disconnected and data saved.")