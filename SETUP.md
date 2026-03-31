# Potato Temp Dashboard — Setup Guide (Raspberry Pi)

## 1. Install dependencies

```bash
pip3 install pandas scikit-learn matplotlib pyserial numpy
```

If on Raspberry Pi OS Bookworm or newer, use:
```bash
pip3 install pandas scikit-learn matplotlib pyserial numpy --break-system-packages
```

## 2. Files needed in the same folder

```
potato_temp_dashboard.py   ← main dashboard script
T_500_seconds.csv          ← your preprocessed training data
```

## 3. Find your Arduino's serial port

Plug in the Arduino, then run:
```bash
python3 potato_temp_dashboard.py --list-ports
```
Common values:
- `/dev/ttyUSB0`  — Arduino Nano/Mega via USB adapter
- `/dev/ttyACM0`  — Arduino Uno/Leonardo

Edit `SERIAL_PORT` at the top of the script to match.

## 4. Give serial port permission (one-time)

```bash
sudo usermod -a -G dialout $USER
# Then log out and back in
```

## 5. Run the dashboard

```bash
# With real Arduino:
python3 potato_temp_dashboard.py

# Test without hardware (demo/simulation mode):
python3 potato_temp_dashboard.py --demo
```

## 6. Key settings to configure (top of script)

| Setting               | Default          | Description                          |
|-----------------------|------------------|--------------------------------------|
| SERIAL_PORT           | /dev/ttyUSB0     | Arduino serial port                  |
| BAUD_RATE             | 9600             | Must match Arduino sketch            |
| TRAINING_CSV          | T_500_seconds.csv| Your preprocessed training data      |
| ALERT_INTERNAL_HIGH   | 35.0 °C          | High temp alert threshold            |
| ALERT_INTERNAL_LOW    | 10.0 °C          | Low temp alert threshold             |
| ACTIVE_MODEL          | random_forest    | Which model prediction to highlight  |
| WINDOW_SIZE           | 100              | Data points shown in chart window    |

## 7. Arduino wiring

- Internal sensor → Analog pin A0
- External sensor → Analog pin A1
- Relay signal    → Digital pin 7
- Sensor type: TMP36 (default) — edit `analogToTemp()` for other sensors

## 8. Run on boot (optional, headless autostart)

Add to `/etc/rc.local` before `exit 0`:
```bash
export DISPLAY=:0
cd /home/pi/potato_temp
python3 potato_temp_dashboard.py &
```
