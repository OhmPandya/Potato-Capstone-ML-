# -*- coding: utf-8 -*-
"""
Potato Temperature Prediction Dashboard
Real-time data collection via Serial/Arduino + ML predictions
Optimized for Raspberry Pi
"""

import sys
import time
import threading
import collections
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import serial
import serial.tools.list_ports

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('TkAgg')  # Works well on Raspberry Pi with desktop
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker

# ──────────────────────────────────────────────
#  CONFIGURATION — edit these to match your setup
# ──────────────────────────────────────────────
SERIAL_PORT     = '/dev/ttyUSB0'   # Change to /dev/ttyACM0 for Arduino Uno
BAUD_RATE       = 9600
SERIAL_TIMEOUT  = 2                # seconds

TRAINING_CSV    = 'T_500_seconds.csv'   # Your preprocessed CSV

ALERT_INTERNAL_HIGH = 35.0        # °C — alert if internal temp exceeds this
ALERT_INTERNAL_LOW  = 10.0        # °C — alert if internal temp drops below this
ALERT_RELAY_OFF     = True        # Alert when relay switches OFF (0)

WINDOW_SIZE     = 100             # Number of recent points shown on chart
POLL_INTERVAL   = 2000            # Chart refresh in ms (matches ~2s sensor rate)

ACTIVE_MODEL    = 'random_forest' # 'linear_regression' or 'random_forest'
# ──────────────────────────────────────────────


# ══════════════════════════════════════════════
#  SERIAL READER  (runs in background thread)
# ══════════════════════════════════════════════
class SerialReader:
    """
    Reads lines from Arduino over serial.
    Expected Arduino output format (one line per ~2 s):
        <time_seconds>,<internal_temp>,<external_temp>,<relay>
    e.g.:  "0.0,23.25,27.50,1"
    """
    def __init__(self, port, baud, timeout):
        self.port    = port
        self.baud    = baud
        self.timeout = timeout
        self.serial  = None
        self.running = False
        self.lock    = threading.Lock()

        # Rolling buffers
        self.times     = collections.deque(maxlen=WINDOW_SIZE)
        self.internal  = collections.deque(maxlen=WINDOW_SIZE)
        self.external  = collections.deque(maxlen=WINDOW_SIZE)
        self.relay     = collections.deque(maxlen=WINDOW_SIZE)
        self.raw_time  = 0.0

        self.last_parse_error = None
        self.connected = False

    def connect(self):
        try:
            self.serial = serial.Serial(
                self.port, self.baud, timeout=self.timeout
            )
            time.sleep(2)  # Let Arduino reset after connect
            self.serial.reset_input_buffer()
            self.connected = True
            print(f"[Serial] Connected → {self.port} @ {self.baud} baud")
            return True
        except serial.SerialException as e:
            print(f"[Serial] Cannot open {self.port}: {e}")
            self.connected = False
            return False

    def _parse_line(self, line: str):
        """Parse CSV line from Arduino."""
        parts = line.strip().split(',')
        if len(parts) < 3:
            raise ValueError(f"Too few fields: {line!r}")
        # Support both 3-field and 4-field formats
        if len(parts) == 3:
            ext_t, int_t, relay = parts
            t = self.raw_time
            self.raw_time += 2.0
        else:
            t, int_t, ext_t, relay = parts[:4]
        return float(t), float(int_t), float(ext_t), int(float(relay))

    def _read_loop(self):
        while self.running:
            try:
                if self.serial and self.serial.in_waiting:
                    raw = self.serial.readline().decode('utf-8', errors='ignore')
                    if raw.strip():
                        t, int_t, ext_t, rel = self._parse_line(raw)
                        with self.lock:
                            self.times.append(t)
                            self.internal.append(int_t)
                            self.external.append(ext_t)
                            self.relay.append(rel)
                        self.last_parse_error = None
                else:
                    time.sleep(0.05)
            except ValueError as e:
                self.last_parse_error = str(e)
            except serial.SerialException as e:
                print(f"[Serial] Read error: {e}")
                self.connected = False
                break

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.serial:
            self.serial.close()

    def snapshot(self):
        """Return a thread-safe copy of current buffers."""
        with self.lock:
            return (
                list(self.times),
                list(self.internal),
                list(self.external),
                list(self.relay),
            )


# ══════════════════════════════════════════════
#  MODEL TRAINER
# ══════════════════════════════════════════════
class TemperatureModels:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.lr  = LinearRegression()
        self.rf  = RandomForestRegressor(max_depth=4, n_estimators=100, random_state=100)
        self.scaler = StandardScaler()
        self.trained = False
        self.metrics = {}

    def train(self):
        try:
            df = pd.read_csv(self.csv_path)
            print(f"[Model] Loaded training data: {len(df)} rows from {self.csv_path}")
        except FileNotFoundError:
            print(f"[Model] WARNING: {self.csv_path} not found — models untrained.")
            return False

        X = df[['External Temperature', 'Time_seconds']].values
        y = df['Internal Temperature'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=100
        )
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        # Linear Regression (on external temp only, matches original script)
        self.lr.fit(X_train_s[:, [0]], y_train)
        lr_pred = self.lr.predict(X_test_s[:, [0]])
        self.metrics['lr'] = {
            'mse': mean_squared_error(y_test, lr_pred),
            'r2':  r2_score(y_test, lr_pred),
        }

        # Random Forest (all features)
        self.rf.fit(X_train_s, y_train)
        rf_pred = self.rf.predict(X_test_s)
        self.metrics['rf'] = {
            'mse': mean_squared_error(y_test, rf_pred),
            'r2':  r2_score(y_test, rf_pred),
        }

        self.trained = True
        print(f"[Model] LR  — MSE: {self.metrics['lr']['mse']:.4f}  R²: {self.metrics['lr']['r2']:.4f}")
        print(f"[Model] RF  — MSE: {self.metrics['rf']['mse']:.4f}  R²: {self.metrics['rf']['r2']:.4f}")
        return True

    def predict(self, external_temp: float, time_s: float) -> dict:
        """Return predictions from both models."""
        if not self.trained:
            return {'lr': None, 'rf': None}
        X = self.scaler.transform([[external_temp, time_s]])
        return {
            'lr': float(self.lr.predict(X[:, [0]])[0]),
            'rf': float(self.rf.predict(X)[0]),
        }


# ══════════════════════════════════════════════
#  ALERT MANAGER
# ══════════════════════════════════════════════
class AlertManager:
    def __init__(self):
        self.active_alerts: list[dict] = []
        self.history: list[dict] = []

    def check(self, int_temp, ext_temp, relay, pred_int):
        new_alerts = []
        ts = time.strftime('%H:%M:%S')

        if int_temp is not None:
            if int_temp > ALERT_INTERNAL_HIGH:
                new_alerts.append({
                    'time': ts, 'level': 'DANGER',
                    'msg': f'Internal temp HIGH: {int_temp:.2f}°C (>{ALERT_INTERNAL_HIGH}°C)'
                })
            if int_temp < ALERT_INTERNAL_LOW:
                new_alerts.append({
                    'time': ts, 'level': 'WARNING',
                    'msg': f'Internal temp LOW: {int_temp:.2f}°C (<{ALERT_INTERNAL_LOW}°C)'
                })

        if ALERT_RELAY_OFF and relay == 0:
            new_alerts.append({
                'time': ts, 'level': 'INFO',
                'msg': f'Relay switched OFF at {ts}'
            })

        if pred_int is not None and int_temp is not None:
            drift = abs(pred_int - int_temp)
            if drift > 3.0:
                new_alerts.append({
                    'time': ts, 'level': 'WARNING',
                    'msg': f'Prediction drift: {drift:.2f}°C (actual={int_temp:.2f}, pred={pred_int:.2f})'
                })

        self.history.extend(new_alerts)
        self.history = self.history[-50:]  # Keep last 50
        self.active_alerts = new_alerts
        return new_alerts


# ══════════════════════════════════════════════
#  DASHBOARD  (Matplotlib figure)
# ══════════════════════════════════════════════
COLORS = {
    'bg':       '#0f1117',
    'panel':    '#1a1d2e',
    'grid':     '#2a2d3e',
    'internal': '#00d4aa',
    'external': '#ff6b6b',
    'pred_lr':  '#ffd166',
    'pred_rf':  '#a78bfa',
    'relay_on': '#06d6a0',
    'relay_off':'#ef476f',
    'text':     '#e8eaf6',
    'subtext':  '#9e9eb0',
    'danger':   '#ef476f',
    'warning':  '#ffd166',
    'info':     '#00d4aa',
}

class Dashboard:
    def __init__(self, reader: SerialReader, models: TemperatureModels,
                 alerts: AlertManager):
        self.reader  = reader
        self.models  = models
        self.alerts  = alerts
        self.last_relay = None

        plt.rcParams.update({
            'font.family':      'monospace',
            'text.color':       COLORS['text'],
            'axes.labelcolor':  COLORS['text'],
            'xtick.color':      COLORS['subtext'],
            'ytick.color':      COLORS['subtext'],
        })

        self.fig = plt.figure(figsize=(14, 8), facecolor=COLORS['bg'])
        self.fig.canvas.manager.set_window_title('🥔 Potato Temp Monitor')

        gs = GridSpec(3, 3, figure=self.fig,
                      hspace=0.45, wspace=0.35,
                      left=0.07, right=0.97, top=0.93, bottom=0.08)

        self.ax_temp   = self.fig.add_subplot(gs[0:2, 0:2])  # main chart
        self.ax_pred   = self.fig.add_subplot(gs[2, 0:2])    # prediction chart
        self.ax_stat   = self.fig.add_subplot(gs[0, 2])      # stats panel
        self.ax_relay  = self.fig.add_subplot(gs[1, 2])      # relay status
        self.ax_alert  = self.fig.add_subplot(gs[2, 2])      # alerts

        self._style_axes()
        self._draw_static_labels()

    def _style_axes(self):
        for ax in [self.ax_temp, self.ax_pred, self.ax_stat,
                   self.ax_relay, self.ax_alert]:
            ax.set_facecolor(COLORS['panel'])
            for spine in ax.spines.values():
                spine.set_color(COLORS['grid'])
            ax.tick_params(colors=COLORS['subtext'], labelsize=8)
            ax.grid(True, color=COLORS['grid'], linewidth=0.5, alpha=0.6)

        for ax in [self.ax_stat, self.ax_relay, self.ax_alert]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)

    def _draw_static_labels(self):
        self.fig.suptitle(
            '🥔  POTATO TEMPERATURE PREDICTION SYSTEM',
            fontsize=13, fontweight='bold',
            color=COLORS['text'], y=0.97
        )
        self.ax_temp.set_title('Live Temperature', color=COLORS['subtext'], fontsize=9)
        self.ax_pred.set_title('Model Predictions vs Actual', color=COLORS['subtext'], fontsize=9)
        self.ax_stat.set_title('Current Readings', color=COLORS['subtext'], fontsize=9)
        self.ax_relay.set_title('Relay Status', color=COLORS['subtext'], fontsize=9)
        self.ax_alert.set_title('Alerts', color=COLORS['subtext'], fontsize=9)

    def _update(self, frame):
        times, internal, external, relay = self.reader.snapshot()
        if not times:
            return

        # ── Latest values ──
        latest_int = internal[-1]
        latest_ext = external[-1]
        latest_rel = relay[-1]
        latest_t   = times[-1]

        preds = self.models.predict(latest_ext, latest_t)
        pred_active = preds[ACTIVE_MODEL[:2]]  # 'lr' or 'rf'

        self.alerts.check(latest_int, latest_ext, latest_rel, pred_active)

        t_arr   = np.array(times)
        int_arr = np.array(internal)
        ext_arr = np.array(external)

        pred_lr_arr = []
        pred_rf_arr = []
        for i in range(len(times)):
            p = self.models.predict(external[i], times[i])
            pred_lr_arr.append(p['lr'] if p['lr'] is not None else np.nan)
            pred_rf_arr.append(p['rf'] if p['rf'] is not None else np.nan)

        # ── Temperature chart ──
        self.ax_temp.cla()
        self.ax_temp.set_facecolor(COLORS['panel'])
        self.ax_temp.grid(True, color=COLORS['grid'], linewidth=0.5, alpha=0.6)
        self.ax_temp.plot(t_arr, int_arr, color=COLORS['internal'],
                          linewidth=2, label='Internal °C')
        self.ax_temp.plot(t_arr, ext_arr, color=COLORS['external'],
                          linewidth=1.5, linestyle='--', label='External °C', alpha=0.8)
        self.ax_temp.axhline(ALERT_INTERNAL_HIGH, color=COLORS['danger'],
                             linewidth=0.8, linestyle=':', alpha=0.7, label=f'High limit ({ALERT_INTERNAL_HIGH}°C)')
        self.ax_temp.axhline(ALERT_INTERNAL_LOW,  color=COLORS['warning'],
                             linewidth=0.8, linestyle=':', alpha=0.7, label=f'Low limit ({ALERT_INTERNAL_LOW}°C)')
        self.ax_temp.set_xlabel('Time (s)', fontsize=8)
        self.ax_temp.set_ylabel('Temperature (°C)', fontsize=8)
        self.ax_temp.set_title('Live Temperature', color=COLORS['subtext'], fontsize=9)
        self.ax_temp.legend(fontsize=7, loc='upper left',
                            facecolor=COLORS['bg'], edgecolor=COLORS['grid'],
                            labelcolor=COLORS['text'])
        self.ax_temp.tick_params(colors=COLORS['subtext'], labelsize=8)
        for sp in self.ax_temp.spines.values():
            sp.set_color(COLORS['grid'])

        # ── Prediction chart ──
        self.ax_pred.cla()
        self.ax_pred.set_facecolor(COLORS['panel'])
        self.ax_pred.grid(True, color=COLORS['grid'], linewidth=0.5, alpha=0.6)
        self.ax_pred.plot(t_arr, int_arr,    color=COLORS['internal'],
                          linewidth=2,   label='Actual Internal °C')
        self.ax_pred.plot(t_arr, pred_lr_arr, color=COLORS['pred_lr'],
                          linewidth=1.5, linestyle='--', label='LR Prediction', alpha=0.85)
        self.ax_pred.plot(t_arr, pred_rf_arr, color=COLORS['pred_rf'],
                          linewidth=1.5, linestyle='-.', label='RF Prediction', alpha=0.85)
        self.ax_pred.set_xlabel('Time (s)', fontsize=8)
        self.ax_pred.set_ylabel('Predicted °C', fontsize=8)
        self.ax_pred.set_title('Model Predictions vs Actual', color=COLORS['subtext'], fontsize=9)
        self.ax_pred.legend(fontsize=7, loc='upper left',
                            facecolor=COLORS['bg'], edgecolor=COLORS['grid'],
                            labelcolor=COLORS['text'])
        self.ax_pred.tick_params(colors=COLORS['subtext'], labelsize=8)
        for sp in self.ax_pred.spines.values():
            sp.set_color(COLORS['grid'])

        # ── Stats panel ──
        self.ax_stat.cla()
        self.ax_stat.set_facecolor(COLORS['panel'])
        self.ax_stat.set_xlim(0, 1); self.ax_stat.set_ylim(0, 1)
        self.ax_stat.axis('off')
        self.ax_stat.set_title('Current Readings', color=COLORS['subtext'], fontsize=9)

        lr_m  = self.models.metrics.get('lr', {})
        rf_m  = self.models.metrics.get('rf', {})
        stats = [
            ('Internal',   f"{latest_int:.2f} °C",  COLORS['internal']),
            ('External',   f"{latest_ext:.2f} °C",  COLORS['external']),
            ('LR Pred',    f"{preds['lr']:.2f} °C" if preds['lr'] else 'N/A', COLORS['pred_lr']),
            ('RF Pred',    f"{preds['rf']:.2f} °C" if preds['rf'] else 'N/A', COLORS['pred_rf']),
            ('LR R²',      f"{lr_m.get('r2', 0):.4f}",  COLORS['pred_lr']),
            ('RF R²',      f"{rf_m.get('r2', 0):.4f}",  COLORS['pred_rf']),
            ('Points',     str(len(times)),           COLORS['subtext']),
            ('Time',       f"{latest_t:.0f} s",       COLORS['subtext']),
        ]
        for i, (label, value, color) in enumerate(stats):
            y = 0.92 - i * 0.115
            self.ax_stat.text(0.04, y, label + ':', fontsize=8,
                              color=COLORS['subtext'], va='top', transform=self.ax_stat.transAxes)
            self.ax_stat.text(0.96, y, value, fontsize=9, fontweight='bold',
                              color=color, va='top', ha='right', transform=self.ax_stat.transAxes)

        # ── Relay panel ──
        self.ax_relay.cla()
        self.ax_relay.set_facecolor(COLORS['panel'])
        self.ax_relay.set_xlim(0, 1); self.ax_relay.set_ylim(0, 1)
        self.ax_relay.axis('off')
        self.ax_relay.set_title('Relay Status', color=COLORS['subtext'], fontsize=9)

        relay_on   = latest_rel == 1
        relay_col  = COLORS['relay_on'] if relay_on else COLORS['relay_off']
        relay_text = 'ON' if relay_on else 'OFF'

        circle = plt.Circle((0.5, 0.52), 0.28, color=relay_col, alpha=0.25)
        self.ax_relay.add_patch(circle)
        circle2 = plt.Circle((0.5, 0.52), 0.20, color=relay_col, alpha=0.6)
        self.ax_relay.add_patch(circle2)
        self.ax_relay.text(0.5, 0.52, relay_text, ha='center', va='center',
                           fontsize=18, fontweight='bold', color=relay_col,
                           transform=self.ax_relay.transAxes)
        # Count relay flips
        flips = sum(1 for i in range(1, len(relay)) if relay[i] != relay[i-1])
        self.ax_relay.text(0.5, 0.12, f'Switches: {flips}', ha='center',
                           fontsize=8, color=COLORS['subtext'],
                           transform=self.ax_relay.transAxes)

        # ── Alert panel ──
        self.ax_alert.cla()
        self.ax_alert.set_facecolor(COLORS['panel'])
        self.ax_alert.set_xlim(0, 1); self.ax_alert.set_ylim(0, 1)
        self.ax_alert.axis('off')
        self.ax_alert.set_title('Alerts', color=COLORS['subtext'], fontsize=9)

        recent = self.alerts.history[-5:][::-1]
        if not recent:
            self.ax_alert.text(0.5, 0.5, 'No alerts', ha='center', va='center',
                               fontsize=9, color=COLORS['subtext'],
                               transform=self.ax_alert.transAxes)
        else:
            level_colors = {
                'DANGER':  COLORS['danger'],
                'WARNING': COLORS['warning'],
                'INFO':    COLORS['info'],
            }
            for i, a in enumerate(recent):
                y = 0.90 - i * 0.19
                col = level_colors.get(a['level'], COLORS['text'])
                self.ax_alert.text(0.04, y, f"[{a['level']}]", fontsize=7,
                                   color=col, fontweight='bold', va='top',
                                   transform=self.ax_alert.transAxes)
                # Wrap long messages
                msg = a['msg']
                if len(msg) > 28:
                    msg = msg[:26] + '…'
                self.ax_alert.text(0.04, y - 0.07, msg, fontsize=6.5,
                                   color=COLORS['text'], va='top',
                                   transform=self.ax_alert.transAxes)

        self.fig.canvas.draw_idle()

    def start(self):
        self.ani = animation.FuncAnimation(
            self.fig, self._update,
            interval=POLL_INTERVAL,
            cache_frame_data=False
        )
        plt.show()


# ══════════════════════════════════════════════
#  DEMO MODE  — simulates Arduino data
# ══════════════════════════════════════════════
class DemoSerialReader(SerialReader):
    """
    Generates synthetic data so you can test the dashboard
    without hardware connected. Run with --demo flag.
    """
    def __init__(self):
        super().__init__(SERIAL_PORT, BAUD_RATE, SERIAL_TIMEOUT)
        self.connected = True
        self._t = 0.0

    def connect(self):
        print("[Demo] Running in DEMO MODE — no serial hardware needed.")
        self.connected = True
        return True

    def _read_loop(self):
        rng = np.random.default_rng(42)
        base_int = 23.0
        base_ext = 27.5
        while self.running:
            noise_int = rng.normal(0, 0.3)
            noise_ext = rng.normal(0, 0.4)
            # Simulate a slow rise
            trend = self._t * 0.005
            int_t = base_int + trend + noise_int
            ext_t = base_ext + noise_ext
            relay = 1 if int_t < 30 else 0

            with self.lock:
                self.times.append(self._t)
                self.internal.append(round(int_t, 2))
                self.external.append(round(ext_t, 2))
                self.relay.append(relay)

            self._t += 2.0
            time.sleep(2.0)


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
def list_serial_ports():
    ports = serial.tools.list_ports.comports()
    if ports:
        print("\nAvailable serial ports:")
        for p in ports:
            print(f"  {p.device}  —  {p.description}")
    else:
        print("No serial ports detected.")

def main():
    demo_mode = '--demo' in sys.argv
    show_ports = '--list-ports' in sys.argv

    if show_ports:
        list_serial_ports()
        return

    print("=" * 55)
    print("  🥔  Potato Temperature Prediction Dashboard")
    print("=" * 55)

    # ── Train models ──
    models = TemperatureModels(TRAINING_CSV)
    models.train()

    # ── Start serial reader ──
    if demo_mode:
        reader = DemoSerialReader()
    else:
        reader = SerialReader(SERIAL_PORT, BAUD_RATE, SERIAL_TIMEOUT)

    if not reader.connect():
        print("\nTip: Run with --demo to test without hardware.")
        print("     Run with --list-ports to see available ports.")
        sys.exit(1)

    reader.start()

    # ── Launch dashboard ──
    alerts    = AlertManager()
    dashboard = Dashboard(reader, models, alerts)

    try:
        dashboard.start()
    except KeyboardInterrupt:
        print("\n[Dashboard] Shutting down...")
    finally:
        reader.stop()
        print("[Dashboard] Stopped.")


if __name__ == '__main__':
    main()
