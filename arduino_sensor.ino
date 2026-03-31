/*
  Potato Temperature Sensor — Arduino Sketch
  ───────────────────────────────────────────
  Reads two temperature sensors and a relay pin,
  then sends CSV over Serial every ~2 seconds.

  Output format:
    <time_seconds>,<internal_temp>,<external_temp>,<relay_state>
  Example:
    0.0,23.25,27.50,1

  Wiring:
    - Internal sensor: DS18B20 or analog TMP36 on A0
    - External sensor: DS18B20 or analog TMP36 on A1
    - Relay signal pin: Digital pin 7 (read)
*/

// ── Pin config ──────────────────────────────
const int PIN_INTERNAL = A0;   // Internal temp sensor (analog)
const int PIN_EXTERNAL = A1;   // External temp sensor (analog)
const int PIN_RELAY    = 7;    // Relay signal (digital input)

const unsigned long INTERVAL_MS = 2000;  // Send every 2 seconds

// ── TMP36 / analog sensor calibration ───────
// If using a different sensor, adjust this function
float analogToTemp(int rawADC) {
  // TMP36: Vout = 10mV/°C offset by 500mV (at 5V supply, 10-bit ADC)
  float voltage = rawADC * (5.0 / 1023.0);
  return (voltage - 0.5) * 100.0;
}

unsigned long lastSend = 0;
float elapsedSeconds   = 0.0;

void setup() {
  Serial.begin(9600);
  pinMode(PIN_RELAY, INPUT);
  delay(500);
  Serial.println("# Potato Temp Monitor — Ready");
}

void loop() {
  unsigned long now = millis();
  if (now - lastSend >= INTERVAL_MS) {
    lastSend = now;

    int rawInternal = analogRead(PIN_INTERNAL);
    int rawExternal = analogRead(PIN_EXTERNAL);
    int relayState  = digitalRead(PIN_RELAY);

    float tempInternal = analogToTemp(rawInternal);
    float tempExternal = analogToTemp(rawExternal);

    // Output: time_seconds, internal, external, relay
    Serial.print(elapsedSeconds, 1);
    Serial.print(",");
    Serial.print(tempInternal, 2);
    Serial.print(",");
    Serial.print(tempExternal, 2);
    Serial.print(",");
    Serial.println(relayState);

    elapsedSeconds += 2.0;
  }
}
