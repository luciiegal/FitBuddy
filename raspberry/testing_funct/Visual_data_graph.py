import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os, sys

# Vérification du port Bluetooth
port = '/dev/rfcomm0'
mac_esp32 = '14:2B:2F:DA:00:CE'
baudrate = 9600

if not os.path.exists(port):
    print("⚠️ Le port /dev/rfcomm0 n'existe pas.")
    print("👉 Essaie : sudo rfcomm bind 0", mac_esp32)
    sys.exit(1)

# Connexion série
ser = serial.Serial(port, baudrate, timeout=1)
time.sleep(2)
print(f"Connexion établie avec {port} à {baudrate} bauds.")
print("Analyse des données avec affichage dynamique...")

# Paramètres
seuil_bas = -8.8
seuil_haut = -8.6
mouvement_vers_haut = False

valeurs_y = deque(maxlen=200)  # Limite pour affichage fluide
amplitudes = []
repetitions = 0
start_time = time.time()

# Configuration du graphe
plt.ion()
fig, (ax_signal, ax_stats) = plt.subplots(2, 1, figsize=(8, 6))

x_data = np.arange(0, len(valeurs_y))
line, = ax_signal.plot([], [], label='Valeur Y', color='b')
ax_signal.set_ylim(-10, 10)
ax_signal.set_xlim(0, 200)
ax_signal.set_xlabel("Échantillons")
ax_signal.set_ylabel("Y")
ax_signal.legend()
ax_signal.grid()

bars = ax_stats.bar(["Répétitions", "Amplitude", "Intensité"], [0, 0, 0], color=['g', 'r', 'orange'])
ax_stats.set_ylim(0, 10)

try:
    while True:
        if ser.in_waiting > 0:
            data = ser.readline().decode().strip()
            if "Accel" in data:
                try:
                    y_value = float(data.split("Y:")[1].split("Z:")[0].strip())
                    valeurs_y.append(y_value)

                    # Détection de mouvement
                    if y_value < seuil_bas and not mouvement_vers_haut:
                        mouvement_vers_haut = True

                    if y_value > seuil_haut and mouvement_vers_haut:
                        repetitions += 1
                        mouvement_vers_haut = False
                        amplitude = max(valeurs_y) - min(valeurs_y)
                        amplitudes.append(amplitude)
                        print(f"🔁 Répétition détectée ({repetitions}) - amplitude {amplitude:.2f}")
                        valeurs_y.clear()

                except ValueError:
                    pass

        # Mise à jour du graphe toutes les 0.2 secondes
        if time.time() - start_time >= 0.2:
            if len(valeurs_y) > 0:
                y_list = list(valeurs_y)
                line.set_data(np.arange(len(y_list)), y_list)
                ax_signal.set_xlim(0, max(200, len(y_list)))
                ax_signal.set_ylim(min(y_list)-1, max(y_list)+1)

                intensite = np.std(y_list)
                amplitude_moy = np.mean(amplitudes) if amplitudes else 0

                # Mise à jour des barres
                bars[0].set_height(repetitions)
                bars[1].set_height(amplitude_moy)
                bars[2].set_height(intensite)

                ax_stats.set_ylim(0, max(10, repetitions + 1))
                fig.suptitle(f"Répétitions: {repetitions} | Amplitude moy: {amplitude_moy:.2f} | Intensité: {intensite:.2f}")

                plt.pause(0.01)
                start_time = time.time()

except KeyboardInterrupt:
    print("Arrêt du programme.")
    ser.close()
    plt.ioff()
    plt.show()
