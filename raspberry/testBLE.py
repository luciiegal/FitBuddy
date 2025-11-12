import serial

# Configuration du port série
port = '/dev/rfcomm0'  # Assure-toi que c'est bien le bon port
baudrate = 9600  # Doit correspondre à la configuration de l'ESP32

try:
    # Ouverture de la connexion série
    ser = serial.Serial(port, baudrate)
    print(f"Connexion établie avec {port} à {baudrate} bauds.\nLecture des données...\n")

    while True:
        if ser.in_waiting > 0:  # Si des données sont disponibles
            data = ser.readline().decode().strip()  # Lecture et décodage des données
            print(f"Données reçues : {data}")

except serial.SerialException:
    print(f"Erreur : Impossible d'ouvrir le port série {port}. Vérifie la connexion.")

except KeyboardInterrupt:
    print("Interruption par l'utilisateur. Fermeture de la connexion.")

finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Connexion série fermée.")

