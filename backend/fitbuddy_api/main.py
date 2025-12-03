from fastapi import FastAPI, HTTPException
from typing import List, Optional
from datetime import datetime
from uuid import UUID

from db import get_connection
from models import (
    Sensor,
    RawSensorData,
    SensorStatus,
    SensorData,
    AsymmetryData,
    Measurement,
    MachineStatus
)


app = FastAPI(
    title="FitBuddy Sensor API",
    description="API pour exposer les données capteurs depuis PostgreSQL",
    version="1.0.0"
)

# --------- HEALTHCHECK ---------

@app.get("/health")
def health_check():
    return {"status": "ok"}


# --------- SENSORS ---------

@app.get("/sensors", response_model=List[Sensor])
def list_sensors():
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM sensor ORDER BY name;")
        rows = cur.fetchall()
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@app.get("/sensors/{sensor_id}", response_model=Sensor)
def get_sensor(sensor_id: UUID):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM sensor WHERE sensor_id = %s;", (str(sensor_id),))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Sensor not found")
        return row
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# --------- RAW SENSOR DATA ---------

@app.get("/sensors/{sensor_id}/raw", response_model=List[RawSensorData])
def get_raw_sensor_data(
    sensor_id: UUID,
    from_ts: Optional[datetime] = None,
    to_ts: Optional[datetime] = None,
    limit: int = 1000
):
    """
    Retourne les données brutes d'un capteur, filtrables par intervalle de temps.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        query = "SELECT * FROM raw_sensor_data WHERE sensor_id = %s"
        params = [str(sensor_id)]

        if from_ts:
            query += " AND timestamp >= %s"
            params.append(from_ts)
        if to_ts:
            query += " AND timestamp <= %s"
            params.append(to_ts)

        query += " ORDER BY timestamp DESC LIMIT %s"
        params.append(limit)

        cur.execute(query, tuple(params))
        rows = cur.fetchall()
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# --------- SENSOR STATUS ---------

@app.get("/sensors/{sensor_id}/status/latest", response_model=SensorStatus)
def get_latest_status(sensor_id: UUID):
    """
    Retourne le dernier statut connu pour un capteur.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM sensor_status
            WHERE sensor_id = %s
            ORDER BY timestamp DESC
            LIMIT 1;
        """, (str(sensor_id),))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="No status found for this sensor")
        return row
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@app.get("/sensors/{sensor_id}/status/history", response_model=List[SensorStatus])
def get_status_history(
    sensor_id: UUID,
    limit: int = 100
):
    """
    Historique des statuts (par défaut les 100 derniers).
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM sensor_status
            WHERE sensor_id = %s
            ORDER BY timestamp DESC
            LIMIT %s;
        """, (str(sensor_id), limit))
        rows = cur.fetchall()
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# --------- SENSOR DATA (reps agrégés) ---------

@app.get("/sensors/{sensor_id}/reps", response_model=List[SensorData])
def get_sensor_reps(
    sensor_id: UUID,
    set_id: Optional[int] = None,
    limit: int = 200
):
    """
    Retourne les données agrégées par rep pour un capteur.
    Option : filtrer par set_id.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        query = "SELECT * FROM sensor_data WHERE sensor_id = %s"
        params = [str(sensor_id)]

        if set_id is not None:
            query += " AND set_id = %s"
            params.append(set_id)

        query += " ORDER BY timestamp DESC, rep_index DESC LIMIT %s"
        params.append(limit)

        cur.execute(query, tuple(params))
        rows = cur.fetchall()
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# --------- ASYMMETRY DATA ---------

@app.get("/asymmetry", response_model=List[AsymmetryData])
def get_asymmetry_data(
    left_sensor_id: Optional[UUID] = None,
    right_sensor_id: Optional[UUID] = None,
    set_id: Optional[int] = None,
    limit: int = 200
):
    """
    Données d'asymétrie gauche/droite.
    Filtres possibles : left_sensor_id, right_sensor_id, set_id.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        query = "SELECT * FROM asymmetry_data WHERE 1=1"
        params = []

        if left_sensor_id:
            query += " AND left_sensor_id = %s"
            params.append(str(left_sensor_id))
        if right_sensor_id:
            query += " AND right_sensor_id = %s"
            params.append(str(right_sensor_id))
        if set_id is not None:
            query += " AND set_id = %s"
            params.append(set_id)

        query += " ORDER BY timestamp DESC, rep_index DESC LIMIT %s"
        params.append(limit)

        cur.execute(query, tuple(params))
        rows = cur.fetchall()
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# --------- MEASUREMENTS (table measurements) ---------

@app.get("/measurements", response_model=List[Measurement])
def get_measurements(limit: int = 100):
    """
    Retourne les derniers messages bruts reçus (table measurements).
    Pour l'instant : simple SELECT topic, payload LIMIT {limit}.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT topic, payload
            FROM measurements
            LIMIT %s;
        """, (limit,))
        rows = cur.fetchall()
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# -----------------------------------------------------
# Machine usage detection
# -----------------------------------------------------
@app.get("/machines/status", response_model=List[MachineStatus])
def machine_status(window_seconds: int = 3, threshold: float = 0.2):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                s.machine,
                MAX(r.timestamp) AS last_activity,
                BOOL_OR(
                    (ABS(COALESCE(r.acc_x,0))
                   + ABS(COALESCE(r.acc_y,0))
                   + ABS(COALESCE(r.acc_z,0))) > %s
                ) AS in_use
            FROM sensor s
            LEFT JOIN raw_sensor_data r
              ON r.sensor_id = s.sensor_id
             AND r.timestamp >= NOW() - (%s || ' seconds')::INTERVAL
            GROUP BY s.machine
            ORDER BY s.machine;
        """, (threshold, str(window_seconds)))
        return cur.fetchall()
    finally:
        conn.close()


@app.get("/machines/{machine}/status", response_model=MachineStatus)
def specific_machine_status(machine: str, window_seconds: int = 3, threshold: float = 0.2):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                s.machine,
                MAX(r.timestamp) AS last_activity,
                BOOL_OR(
                    (ABS(COALESCE(r.acc_x,0))
                   + ABS(COALESCE(r.acc_y,0))
                   + ABS(COALESCE(r.acc_z,0))) > %s
                ) AS in_use
            FROM sensor s
            LEFT JOIN raw_sensor_data r
              ON r.sensor_id = s.sensor_id
             AND r.timestamp >= NOW() - (%s || ' seconds')::INTERVAL
            WHERE s.machine=%s
            GROUP BY s.machine;
        """, (threshold, str(window_seconds), machine))
        
        res = cur.fetchone()
        if not res:
            raise HTTPException(404, "Machine not found")
        return res

    finally:
        conn.close()
