import time, numpy as np, matplotlib.pyplot as plt, csv
from collections import deque
import os, sys

# --- serial import (safe for SIMULATE mode) ---
try:
    import serial  # used when SIMULATE = False
except Exception:
    serial = None

# ---------- SIMULATION OR REAL DATA ----------
SIMULATE = False   # True = simulate / False = ESP32 serial

# ---------- SERIAL CONNECTION ----------
port = '/dev/rfcomm0'
mac_esp32 = '14:2B:2F:DA:00:CE'
baudrate = 9600

# ---------- PARAMETERS ----------
SEUIL_LOW  = -8.8
SEUIL_HIGH = -8.6
ROLLING_BASELINE = 50
SMOOTH_WINDOW    = 5
MAX_BUFFER       = 600
mass_kg = 20.0
g = 9.81

PREBUFFER_SAMPLES   = 30
MIN_REP_DURATION    = 0.35
MIN_PEAK_TO_PEAK_A  = 0.60

# ---------- CREATE SESSION CSV ----------
os.makedirs("data", exist_ok=True)
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
csv_filename = os.path.join("data", f"session_{timestamp}.csv")
csv_fields = [
    "Rep", "Force_min", "Force_max", "Force_mean",
    "Power_min", "Power_max", "Power_mean",
    "Tempo_C", "Tempo_E", "F/V_ratio"
]
csv_file = open(csv_filename, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(csv_fields)
csv_file.flush()
print("📁 CSV will be saved to:", os.path.abspath(csv_filename))

# ---------- UTILITIES ----------
def rolling_mean(x, w):
    if w <= 1 or len(x) < w:
        return np.array(x, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[w:] - c[:-w]) / float(w)
    pad = np.concatenate([np.full(w-1, out[0]), out])
    return pad

def process_signal(t, y):
    y = np.array(y, dtype=float)
    t = np.array(t, dtype=float)
    base = rolling_mean(y, ROLLING_BASELINE)
    a_dyn = y - base
    if SMOOTH_WINDOW > 1:
        a_dyn = rolling_mean(a_dyn, SMOOTH_WINDOW)
    if len(t) < 2:
        return a_dyn, np.zeros_like(a_dyn)
    dt = np.diff(t, prepend=t[0])
    dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 1/50.0
    v = np.cumsum(a_dyn * dt)
    v -= rolling_mean(v, ROLLING_BASELINE)
    return a_dyn, v

def summarize_rep(t_rep, a_rep, v_rep):
    t_rep = np.array(t_rep); a_rep = np.array(a_rep); v_rep = np.array(v_rep)
    if len(t_rep) < 2:
        return None
    duration = float(t_rep[-1] - t_rep[0])
    if duration < MIN_REP_DURATION:
        return None
    if (np.max(a_rep) - np.min(a_rep)) < MIN_PEAK_TO_PEAK_A:
        return None

    # --- Force & Power ---
    F_signed = mass_kg * a_rep      # physical force (signed)
    F = np.abs(F_signed)            # report force as positive
    P = F_signed * v_rep            # power keeps physical sign

    # --- Tempo concentrique / excentrique ---
    dt = np.diff(t_rep, prepend=t_rep[0])
    conc = v_rep >= 0
    ecc  = v_rep < 0
    tempo_con = float(np.sum(dt[conc]))
    tempo_ecc = float(np.sum(dt[ecc]))

    # --- Force / Velocity ratio ---
    mean_abs_v = float(np.mean(np.abs(v_rep))) if np.any(np.abs(v_rep) > 0) else 0.0
    fv_ratio = float(np.mean(F)) / mean_abs_v if mean_abs_v > 1e-6 else 0.0

    return dict(
        force_min=float(np.min(F)),
        force_max=float(np.max(F)),
        force_mean=float(np.mean(F)),
        power_min=float(np.min(P)),
        power_max=float(np.max(P)),
        power_mean=float(np.mean(P)),
        tempo_con=tempo_con,
        tempo_ecc=tempo_ecc,
        fv_ratio=fv_ratio
    )

def session_aggregates(summaries):
    """
    Aggregate ALL reps to get:
    - Force min / max / mean
    - Power min / max / mean
    - Mean concentric / eccentric tempo
    - Mean F/V ratio
    """
    if not summaries:
        return {
            "force_min": 0.0, "force_max": 0.0, "force_mean": 0.0,
            "power_min": 0.0, "power_max": 0.0, "power_mean": 0.0,
            "tempo_con_mean": 0.0, "tempo_ecc_mean": 0.0,
            "fv_ratio_mean": 0.0
        }

    f_min  = [s["force_min"]   for s in summaries]
    f_max  = [s["force_max"]   for s in summaries]
    f_mean = [s["force_mean"]  for s in summaries]
    p_min  = [s["power_min"]   for s in summaries]
    p_max  = [s["power_max"]   for s in summaries]
    p_mean = [s["power_mean"]  for s in summaries]
    t_con  = [s["tempo_con"]   for s in summaries]
    t_ecc  = [s["tempo_ecc"]   for s in summaries]
    fv_all = [s["fv_ratio"]    for s in summaries]

    return {
        "force_min":  float(np.min(f_min)),
        "force_max":  float(np.max(f_max)),
        "force_mean": float(np.mean(f_mean)),
        "power_min":  float(np.min(p_min)),
        "power_max":  float(np.max(p_max)),
        "power_mean": float(np.mean(p_mean)),
        "tempo_con_mean": float(np.mean(t_con)),
        "tempo_ecc_mean": float(np.mean(t_ecc)),
        "fv_ratio_mean":  float(np.mean(fv_all))
    }

def save_to_csv(rep_id, summary):
    csv_writer.writerow([
        rep_id,
        summary["force_min"], summary["force_max"], summary["force_mean"],
        summary["power_min"], summary["power_max"], summary["power_mean"],
        summary["tempo_con"], summary["tempo_ecc"], summary["fv_ratio"]
    ])
    csv_file.flush()

# ---------- SIMULATED STREAM ----------
def simulated_stream(fps=50, reps=8, up_time=0.9, down_time=1.0, pause=0.4,
                     amp=2.2, noise_std=0.08, drift=0.0):
    period = up_time + down_time + pause
    total_time = reps * period
    t0 = time.time()
    t = 0.0
    while t < total_time:
        phase_t = t % period
        if phase_t < up_time:
            a_dyn = amp * np.sin(np.pi * phase_t / up_time)
        elif phase_t < up_time + down_time:
            tt = phase_t - up_time
            a_dyn = -amp * np.sin(np.pi * tt / down_time)
        else:
            a_dyn = 0.0
        y = -g + a_dyn + np.random.normal(0, noise_std) + drift*(t/total_time)
        x = np.random.normal(0, noise_std*0.2)
        z = np.random.normal(0, noise_std*0.2)
        yield f"Accel X: {x:.3f} Y: {y:.3f} Z: {z:.3f}\n"
        time.sleep(1.0/fps)
        t = time.time() - t0

# ---------- DATA SOURCE ----------
def line_source():
    if SIMULATE:
        for line in simulated_stream():
            yield line
    else:
        if serial is None:
            raise RuntimeError("pyserial not available. Install with: pip install pyserial")
        if not os.path.exists(port):
            print("⚠ Port /dev/rfcomm0 not found. Try: sudo rfcomm bind 0", mac_esp32)
            sys.exit(1)
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)
        print(f"Connected to {port} @ {baudrate}")
        try:
            while True:
                if ser.in_waiting > 0:
                    yield ser.readline().decode(errors='ignore')
        finally:
            ser.close()

# ---------- MAIN ----------
def main():
    print("Running:", "SIMULATION mode" if SIMULATE else "REAL SERIAL mode")

    y_raw, t_raw = deque(maxlen=MAX_BUFFER), deque(maxlen=MAX_BUFFER)
    prebuf_y, prebuf_t = deque(maxlen=PREBUFFER_SAMPLES), deque(maxlen=PREBUFFER_SAMPLES)
    rep_y_raw, rep_t_raw = [], []

    # --- Layout: Signals on top (full width); bottom row has 3 horizontal bars ---
    plt.ion()
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1.2])

    # Top (row 0, spans all 3 columns): signals
    ax_signal = fig.add_subplot(gs[0, :])
    line_raw,   = ax_signal.plot([], [], label='Y raw', lw=1)
    line_debias,= ax_signal.plot([], [], label='a_dyn (no gravity)', lw=1)
    line_vel,   = ax_signal.plot([], [], label='velocity (integrated)', lw=1)
    ax_signal.legend(); ax_signal.grid(True); ax_signal.set_xlabel("Samples")

    # Bottom row (row 1): three separate bar charts side-by-side
    ax_reps  = fig.add_subplot(gs[1, 0])
    ax_force = fig.add_subplot(gs[1, 1])
    ax_power = fig.add_subplot(gs[1, 2])

    bar_reps  = ax_reps.bar(["Reps"], [0])
    bar_force = ax_force.bar(["Fmin", "Fmean", "Fmax"], [0, 0, 0])
    bar_power = ax_power.bar(["Pmin", "Pmean", "Pmax"], [0, 0, 0])

    ax_reps.set_ylabel("Count")
    ax_force.set_ylabel("Force (N)")
    ax_power.set_ylabel("Power (W)")

    for ax in (ax_reps, ax_force, ax_power):
        ax.grid(True, axis='y')

    moving_up, repetitions = False, 0
    rep_summaries = []
    start_time = time.time()

    for data in line_source():
        if "Accel" in data and "Y:" in data:
            try:
                y_value = float(data.split("Y:")[1].split("Z:")[0].strip())
            except:
                continue
            now = time.time()

            # live buffers + prebuffer
            y_raw.append(y_value); t_raw.append(now)
            prebuf_y.append(y_value); prebuf_t.append(now)

            # start
            if (y_value < SEUIL_LOW) and (not moving_up):
                moving_up = True
                rep_y_raw = list(prebuf_y)
                rep_t_raw = list(prebuf_t)

            # accumulate while active
            if moving_up:
                rep_y_raw.append(y_value)
                rep_t_raw.append(now)

            # end
            if (y_value > SEUIL_HIGH) and moving_up:
                moving_up = False
                a_seg, v_seg = process_signal(rep_t_raw, rep_y_raw)
                summary = summarize_rep(rep_t_raw, a_seg, v_seg)
                if summary:
                    repetitions += 1
                    rep_summaries.append(summary)
                    save_to_csv(repetitions, summary)
                    print(
                        f"🔁 Rep {repetitions:02d} | "
                        f"F[N] min/max/mean: {summary['force_min']:.1f}/"
                        f"{summary['force_max']:.1f}/{summary['force_mean']:.1f} | "
                        f"P[W] min/max/mean: {summary['power_min']:.1f}/"
                        f"{summary['power_max']:.1f}/{summary['power_mean']:.1f} | "
                        f"Tempo C/E[s]: {summary['tempo_con']:.2f}/"
                        f"{summary['tempo_ecc']:.2f} | "
                        f"F/V ratio: {summary['fv_ratio']:.2f}"
                    )
                rep_y_raw, rep_t_raw = [], []

        # refresh every 0.2 s
        if time.time() - start_time >= 0.2 and len(y_raw) > 5:
            a_all, v_all = process_signal(list(t_raw), list(y_raw))

            x = np.arange(len(y_raw))
            line_raw.set_data(x, list(y_raw))
            line_debias.set_data(x, a_all)
            line_vel.set_data(x, v_all)

            ymin = min(np.min(y_raw), np.min(a_all), np.min(v_all))
            ymax = max(np.max(y_raw), np.max(a_all), np.max(v_all))
            ax_signal.set_xlim(0, max(200, len(y_raw)))
            ax_signal.set_ylim(ymin-1, ymax+1)

            sess = session_aggregates(rep_summaries)

            # --- update bar charts ---

            # Reps
            bar_reps[0].set_height(repetitions)
            ax_reps.set_ylim(0, max(1, repetitions + 1))

            # Force: Fmin / Fmean / Fmax
            fmin, fmean, fmax = sess["force_min"], sess["force_mean"], sess["force_max"]
            bar_force[0].set_height(fmin)
            bar_force[1].set_height(fmean)
            bar_force[2].set_height(fmax)
            ax_force.set_ylim(0, max(1.0, fmax * 1.3))

            # Power: Pmin / Pmean / Pmax （允许负值，显示 0 线）
            pmin, pmean, pmax = sess["power_min"], sess["power_mean"], sess["power_max"]
            bar_power[0].set_height(pmin)
            bar_power[1].set_height(pmean)
            bar_power[2].set_height(pmax)

            p_top = max(pmax, 0.0)
            p_bottom = min(pmin, 0.0)
            if abs(p_top - p_bottom) < 1e-3:  # 避免范围太小
                p_top += 1.0
                p_bottom -= 1.0
            ax_power.set_ylim(p_bottom * 1.3, p_top * 1.3)
            ax_power.axhline(0, linewidth=0.8)

            # --- Title: 一行 F/P, 一行 Tempo + F/V ---
            if rep_summaries:
                fig.suptitle(
                    "Reps: {} | F[min/mean/max]={:.1f}/{:.1f}/{:.1f} N | "
                    "P[min/mean/max]={:.1f}/{:.1f}/{:.1f} W\n"
                    "⟨Tempo⟩ C/E={:.2f}/{:.2f} s | ⟨F/V⟩={:.2f}".format(
                        repetitions,
                        sess["force_min"], sess["force_mean"], sess["force_max"],
                        sess["power_min"], sess["power_mean"], sess["power_max"],
                        sess["tempo_con_mean"], sess["tempo_ecc_mean"],
                        sess["fv_ratio_mean"]
                    )
                )
            else:
                fig.suptitle(f"Reps: {repetitions}")

            plt.pause(0.01)
            start_time = time.time()

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    finally:
        try:
            csv_file.close()
            print("✅ Data saved to:", os.path.abspath(csv_filename))
        except Exception:
            pass





