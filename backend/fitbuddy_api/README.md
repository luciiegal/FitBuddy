# 📘 **README – FitBuddy Sensor API**

## 🏋️ Overview

FitBuddy Sensor API est une API développée avec **FastAPI** permettant d’exposer et analyser les données issues de capteurs installés sur des machines de musculation.  
Les capteurs envoient leurs données via **MQTT**, un script d’ingestion les insère dans **PostgreSQL**, puis l’API rend ces données accessibles à diverses applications (mobile, web, dashboards…).

Les objectifs principaux sont :
- déterminer en temps réel si une machine est utilisée ou non,
- analyser les répétitions (reps) et séries (sets),
- détecter la fin d’une série,
- fournir des données d’asymétrie,
- exposer des données fiables via une API REST.

---

## 🚀 Features

* ✔️ Liste des capteurs installés  
* ✔️ Données brutes (accéléromètre + gyroscope)  
* ✔️ Statut des capteurs (batterie, température, signal…)  
* ✔️ Données agrégées par rep (analyse du mouvement)  
* ✔️ Détection de machine utilisée / non utilisée  
* ✔️ Détection de fin de série  
* ✔️ Données d’asymétrie gauche/droite  
* ✔️ Documentation interactive via Swagger UI  

---

## 🧱 Architecture générale

Raspberry Pi  →  Mosquitto MQTT  
                     ↓  
        collector_db.py (ingestion)  
                     ↓  
        PostgreSQL (FitBuddyDB)  
                     ↓  
              FastAPI backend  
                     ↓  
 Application mobile / Dashboard / Scripts  

---

## 📦 Installation

### 1️⃣ Prérequis

Assurez-vous d’avoir installé :

- Python 3.10+  
- pip  
- Git  
- PostgreSQL

### 2️⃣ Cloner le repository

```bash
git clone https://github.com/em-madurand/FitBuddy.git
cd FitBuddy
```

### 3️⃣ Créer un environnement virtuel

Windows PowerShell :

```powershell
py -3 -m venv venv
.
env\Scripts\Activate.ps1
```

Linux / macOS :

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4️⃣ Installer les dépendances Python

```bash
pip install fastapi uvicorn psycopg2-binary python-dotenv
```

### 5️⃣ Configurer les variables d’environnement

Créer un fichier `.env` à la racine :

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=FitBuddyDB
DB_USER=postgres
DB_PASSWORD=ESILVPI2projet?
```

### 6️⃣ Lancer le serveur FastAPI

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI :  
http://localhost:8000/docs

---

## 📘 Models (Pydantic)

### `Sensor`

```python
sensor_id: UUID
name: str
machine: str
position: str
installed_at: datetime
is_active: bool
```

### `RawSensorData`

```python
id: int
timestamp: datetime
sensor_id: UUID
acc_x, acc_y, acc_z
gyro_x, gyro_y, gyro_z
```

### `SensorStatus`

```python
battery_level: int
temperature: float
signal_strength: int
storage_free: int
hours_used: float
status: str
```

### `SensorData` (reps)

```python
set_id: int
rep_index: int
rep_duration: float
rest_time_before: float
speed_concentric: float
speed_eccentric: float
amplitude: float
difficulty_level: int
```

### `AsymmetryData`

```python
left_sensor_id: UUID
right_sensor_id: UUID
amplitude_diff: float
speed_diff: float
asymmetry_score: float
```

### `Measurement`

```python
topic: str
payload: str
```

---

## 🌐 Endpoints

### 🩺 Healthcheck

**GET /health**

### 🧩 Capteurs

**GET /sensors**  
**GET /sensors/{sensor_id}**

### 📊 Données brutes

**GET /sensors/{sensor_id}/raw**

Params : `from_ts`, `to_ts`, `limit`

### 🔋 Statut capteurs

**GET /sensors/{sensor_id}/status/latest**  
**GET /sensors/{sensor_id}/status/history**

### 🏋️ Reps

**GET /sensors/{sensor_id}/reps**

### ⚖️ Asymétrie

**GET /asymmetry**

### 🔥 Détection machine utilisée

**GET /machines/status**  
**GET /machines/{machine}/status**

---

## 🧠 Détection de fin de série

Une série est terminée lorsque :

```
NOW() - timestamp_derniere_rep > 100s
```

Exemple JSON :

```json
{
  "sensor_id": "S1",
  "machine": "Leg Press",
  "set_id": 7,
  "rep_count": 3,
  "start_time": "2025-12-03T10:15:12.345000",
  "end_time": "2025-12-03T10:15:17.900000",
  "avg_rep_duration": 1.7,
  "avg_amplitude": 0.95,
  "status": "finished"
}
```

---

## 📚 Conclusion

Cette API permet la détection en temps réel de l’utilisation des machines, l’analyse des répétitions, la détection des fins de séries et l’accès aux données nécessaires à une application de salle de sport moderne.
