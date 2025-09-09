
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from models import *
from orbital import AtlasTracker
from mission import MissionPlanner
import math

app = FastAPI(
    title="31/ATLAS Mission Control",
    description="Intercept the mysterious interstellar visitor!",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])


atlas_tracker = AtlasTracker()
mission_planner = MissionPlanner()

@app.get("/")
async def root():
    return {"message": "🛸 31/ATLAS Mission Control Online", "status": "tracking"}

@app.get("/atlas/status", response_model=AtlasStatus)
async def get_atlas_status():
    try:
        pos = atlas_tracker.get_current_position()
        speed = atlas_tracker.calculate_speed(datetime.now())
        distance_earth = atlas_tracker.distance_from_earth(pos, datetime.now())
        distance_sun = math.sqrt(pos.x**2 + pos.y**2 + pos.z**2)
        
        return AtlasStatus(
            position=pos,
            velocity=speed * 3600,  
            dist_from_earth_au=distance_earth,
            dist_from_sun_au=distance_sun,
            last_updated=datetime.now()
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        # print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mission/plan", response_model=MissionResult)
async def plan_mission(request: MissionRequest):
    
    try:
        result = mission_planner.plan_mission(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


MU_SUN = 1.32712440018e11   # km^3/s^2
AU_KM = 1.495978707e8       # km
CRUISE_SPEED = 15.0         # km/s, assumed spacecraft cruise velocity

@app.get("/mission/optimal-windows")
async def get_launch_windows():
    windows = []
    base_date = datetime.now()
    metrics = []

    for days in range(0, 365, 30):
        window_date = base_date + timedelta(days=days)

        # Atlas position
        atlas_pos = atlas_tracker.get_position_at_date(window_date)
        atlas_distance_au = math.sqrt(atlas_pos.x**2 + atlas_pos.y**2 + atlas_pos.z**2)
        r1 = 1.0 * AU_KM   # Earth orbit
        r2 = atlas_distance_au * AU_KM

        # Semi-major axis for transfer ellipse
        a_transfer = (r1 + r2) / 2.0

        # Velocities
        v_earth = math.sqrt(MU_SUN / r1)
        v_atlas = math.sqrt(MU_SUN / r2)

        v_perihelion = math.sqrt(MU_SUN * (2/r1 - 1/a_transfer))
        v_aphelion   = math.sqrt(MU_SUN * (2/r2 - 1/a_transfer))

        # Δv budget
        delta_v1 = abs(v_perihelion - v_earth)
        delta_v2 = abs(v_atlas - v_aphelion)
        delta_v = delta_v1 + delta_v2

        # ToF (approx by cruise speed)
        distance_km = r2 - r1
        tof_days = distance_km / CRUISE_SPEED / 86400  # seconds → days

        metrics.append((delta_v, tof_days))

        windows.append({
            "date": window_date.strftime("%Y-%m-%d"),
            "delta_v": delta_v,
            "tof_days": tof_days,
            "atlas_distance_au": round(atlas_distance_au, 3)
        })

    # Normalization for scoring
    dv_min, dv_max = min(m[0] for m in metrics), max(m[0] for m in metrics)
    tof_min, tof_max = min(m[1] for m in metrics), max(m[1] for m in metrics)

    scored_windows = []
    for idx, w in enumerate(windows):
        dv, tof = w["delta_v"], w["tof_days"]

        # Normalize (lower = better)
        dv_score = (dv_max - dv) / (dv_max - dv_min + 1e-6)
        tof_score = (tof_max - tof) / (tof_max - tof_min + 1e-6)

        # Weighted score (Δv more important than ToF, e.g. 60/40)
        raw_score = 0.6 * dv_score + 0.4 * tof_score

        # Scale 0–9.5 instead of 0–10
        final_score = round(raw_score * 9.5, 1)

        scored_windows.append({
            "date": w["date"],
            "score": final_score,
            "delta_v_kms": round(dv, 2),
            "transfer_time_years": round(w["tof_days"] / 365.25, 1),
            "atlas_distance_au": w["atlas_distance_au"],
            "description": f"Efficiency {final_score:.1f}/10 | Δv={dv:.2f} km/s | ToF={w['tof_days']/365.25:.1f} years"
        })

    return sorted(scored_windows, key=lambda x: x["score"], reverse=True)[:6]
