import serial
import time
import numpy as np
from final import *

# Configuration du port série (déjà connecté en rfcomm)
port = '/dev/rfcomm0'
baudrate = 9600
ser = serial.Serial(port, baudrate, timeout=1)
time.sleep(2)

print("Connexion série établie avec", port)

def parse_imu_message(msg):
    """
    Parse les données envoyées par l'ESP32.
    Retourne un dict {"ax":..., "gx":..., etc}
    """
    data = {}

    try:
        lines = msg.split("\n")
        for line in lines:

            # Accélération
            if "Accel" in line:
                # Ex: "🔹 Accel (m/s²) → X: -0.12 Y: 9.81 Z: 0.03"
                parts = line.split("→")[1].split()
                data["ax"] = float(parts[1])
                data["ay"] = float(parts[3])
                data["az"] = float(parts[5])

            # Gyroscope
            elif "Gyro" in line:
                parts = line.split("→")[1].split()
                data["gx"] = float(parts[1])
                data["gy"] = float(parts[3])
                data["gz"] = float(parts[5])
    
    except Exception as e:
        print("⚠ Parsing error :", e)

    return data


buffer = ""  # Pour accumuler les messages multi-lignes

# --- Kalman Filter ---
kf = KalmanFilterAccelerometer(
    process_noise=0.003,
    measurement_noise=0.1
)

# --- Points ---
filtered_history = []
WINDOW_SIZE = 10         # nombre de points à considérer pour détecter la fin
STOP_THRESHOLD = 0.9  # variation minimale pour considérer le mouvement arrêté
timer = []
print("Start")

while True:
    try:
        line = ser.readline().decode("utf-8", errors="ignore")

        if not line:
            continue

        buffer += line

        # Le message complet envoyé par l'ESP32 se termine par un saut de ligne vide
        if line.strip() == "":
            #print("Message brut reçu :")
            #print(buffer)

            imu_data = parse_imu_message(buffer)

            if imu_data:
                #print("Données IMU parsées :", imu_data)

                ax = imu_data.get("ax")
                ay = imu_data.get("ay")
                az = imu_data.get("az")

                # Si on a les données d'accélération
                if ax is not None and ay is not None and az is not None:

                    # --- Kalman ---
                    #kf.predict()
                    #filtered = kf.update([ax, ay, az])

                    #filtered_history.append(filtered)
                    timer.append(time.time())
                    filtered_history.append([ax, ay, az])

                    #print("Kalman filtré :", filtered)
                    #if len(filtered_history) > 2000:
                        #filtered_history.pop(0)
                    # Vérifier si l’exo est terminé : variation sur les 5 derniers points < seuil
                    
                if len(filtered_history) >= WINDOW_SIZE:
                   
                    #print("rw")
                    recent_window = np.array(filtered_history[-WINDOW_SIZE:])
                    #print(recent_window)
                    axis_data = recent_window[:, 2]  # axe Y par exemple, à adapter

                    #print(axis_data)
                    variation = np.max(axis_data) - np.min(axis_data)
                    print(variation)

                    if variation < STOP_THRESHOLD:
                        # On considère que le mouvement est fini
                        arr = np.array(filtered_history)

                        save_workout_csv(timer, filtered_history, filename='test.csv')
                        # Optionnel : arrêter la boucle ou réinitialiser
                        break  # arrêter la lecture série



            buffer = ""  # reset pour le prochain message

    except KeyboardInterrupt:
        save_workout_csv(timer, filtered_history, filename=None)
        print("\n Arrêt du programme.")
        break
