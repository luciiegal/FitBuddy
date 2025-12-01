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
