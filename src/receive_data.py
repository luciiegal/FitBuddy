
import serial
import time
import numpy as np

# Configuration du port série
port = '/dev/rfcomm0'  # Assure-toi que c'est le bon port
baudrate = 9600
ser = serial.Serial(port, baudrate, timeout=1)
time.sleep(2)  # Attente pour la stabilisation de la connexion

print("Connexion établie avec", port, "à", baudrate, "bauds.")
print("Analyse des données...")

# Variables pour les statistiques
repetitions = 0
seuil_bas = -8.8  # Seuil bas pour détecter la position basse du mouvement
seuil_haut = -8.6  # Seuil haut pour détecter la position haute du mouvement
mouvement_vers_haut = False

valeurs_y = []  # Liste pour enregistrer les valeurs de l'axe Y
amplitudes = []  # Amplitudes des répétitions
start_time = time.time()

try:
    while True:
        if ser.in_waiting > 0:
            data = ser.readline().decode().strip()
            
            if "Accel" in data:
                try:
                    # Extraction de la valeur Y
                    y_value = float(data.split("Y:")[1].split("Z:")[0].strip())
                    valeurs_y.append(y_value)

                    # Comptage des répétitions
                    if y_value < seuil_bas and not mouvement_vers_haut:
                        mouvement_vers_haut = True

                    if y_value > seuil_haut and mouvement_vers_haut:
                        repetitions += 1
                        mouvement_vers_haut = False

                        # Calcul de l'amplitude (différence max-min)
                        amplitude = max(valeurs_y) - min(valeurs_y)
                        amplitudes.append(amplitude)

                        print(f"🔁 Répétition détectée ! Total : {repetitions}")
                        print(f"📏 Amplitude de la répétition : {amplitude:.2f}")

                        # Réinitialisation des valeurs pour la prochaine répétition
                        valeurs_y.clear()

                except ValueError:
                    print("Erreur de conversion des données.")

            # Affichage des statistiques toutes les 10 secondes
            if time.time() - start_time >= 10:
                if len(valeurs_y) > 0:
                    # Calcul de l'intensité (écart-type) et de l'asymétrie (moyenne des valeurs absolues)
                    intensite = np.std(valeurs_y)
                    asymetrie = np.mean(np.abs(valeurs_y))

                    print("\n--- Statistiques ---")
                    print(f"🔁 Répétitions totales : {repetitions}")
                    print(f"📏 Amplitude moyenne : {np.mean(amplitudes):.2f}")
                    print(f"📊 Intensité (écart-type) : {intensite:.2f}")
                    print(f"⚖️ Asymétrie (moyenne des abs) : {asymetrie:.2f}\n")

                    # Réinitialisation du temps pour la prochaine fenêtre de stats
                    start_time = time.time()

except KeyboardInterrupt:
    print("Arrêt du programme.")
    ser.close()
    
##14:2B:2F:DA:00:CE (adresse mac de l'esp32)
