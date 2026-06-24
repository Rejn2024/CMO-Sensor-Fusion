import os
import time
import csv

# Update these paths to match your CMO installation directory
CMO_LOG_PATH = r"C:\Command Modern Operations\Logs\LuaHistory.txt"
OUTPUT_CSV = "cmo_sensor_logs.csv"

# Ensure output CSV has a header row if it doesn't exist
if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Timestamp", "Observer_Aircraft", "Contact_Name", "Contact_Type", "Classification", "Age_Seconds"])

print("Watching CMO Log File for sensor updates... Press Ctrl+C to stop.")


def watch_log():
    # Open the file and jump straight to the current end so we only read new lines
    with open(CMO_LOG_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        f.seek(0, os.SEEK_END)

        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)  # Wait briefly for new simulation ticks
                continue

            # Filter specifically for our Lua script prefix
            if "PY_CONTACT_LOG" in line:
                # Strip clean and split by comma
                clean_line = line.strip().split("PY_CONTACT_LOG,")[-1]
                data_fields = clean_line.split(",")

                # Append the clean data row straight to your CSV
                with open(OUTPUT_CSV, mode='a', newline='', encoding='utf-8') as out_f:
                    csv.writer(out_f).writerow(data_fields)

                print(f"Logged: {data_fields[0]} | {data_fields[1]} sees {data_fields[2]}")


if __name__ == "__main__":
    try:
        watch_log()
    except KeyboardInterrupt:
        print("\nLogging stopped successfully.")
