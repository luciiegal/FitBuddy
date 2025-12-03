from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from uuid import UUID

# -------------------------
# 1. CAPTEURS (table sensor)
# -------------------------
class Sensor(BaseModel):
    sensor_id: UUID
    name: Optional[str] = None
    machine: Optional[str] = None
    position: Optional[str] = None
    installed_at: Optional[datetime] = None
    is_active: Optional[bool] = None


# -----------------------------------
# 2. DONNÉES BRUTES (raw_sensor_data)
# -----------------------------------
class RawSensorData(BaseModel):
    id: int
    timestamp: datetime
    sensor_id: UUID
    acc_x: Optional[float] = None
    acc_y: Optional[float] = None
    acc_z: Optional[float] = None
    gyro_x: Optional[float] = None
    gyro_y: Optional[float] = None
    gyro_z: Optional[float] = None


# ------------------------------
# 3. STATUTS DES CAPTEURS (sensor_status)
# ------------------------------
class SensorStatus(BaseModel):
    id: int
    timestamp: datetime
    sensor_id: UUID
    battery_level: Optional[int] = None
    temperature: Optional[float] = None
    signal_strength: Optional[int] = None
    storage_free: Optional[int] = None
    hours_used: Optional[float] = None
    status: Optional[str] = None


# --------------------------------
# 4. DONNÉES AGRÉGÉES (sensor_data)
# --------------------------------
class SensorData(BaseModel):
    id: int
    timestamp: datetime
    sensor_id: UUID
    set_id: Optional[int] = None
    rep_index: Optional[int] = None
    rep_duration: Optional[float] = None
    rest_time_before: Optional[float] = None
    speed_concentric: Optional[float] = None
    speed_eccentric: Optional[float] = None
    amplitude: Optional[float] = None
    difficulty_level: Optional[int] = None


# -------------------------------------
# 5. ASYMÉTRIE (asymmetry_data)
# -------------------------------------
class AsymmetryData(BaseModel):
    id: int
    timestamp: datetime
    left_sensor_id: Optional[UUID] = None
    right_sensor_id: Optional[UUID] = None
    set_id: Optional[int] = None
    rep_index: Optional[int] = None
    amplitude_diff: Optional[float] = None
    speed_diff: Optional[float] = None
    asymmetry_score: Optional[float] = None

# 6. Mesures brutes MQTT (table measurements)
class Measurement(BaseModel):
    topic: str
    payload: str

# 7. Machine status (new)
class MachineStatus(BaseModel):
    machine: str
    in_use: bool
    last_activity: Optional[datetime]

