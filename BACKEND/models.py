from pydantic import BaseModel,Field
from typing import List, Dict
from enum import Enum
from datetime import datetime

class PropulsionType(str,Enum):
    CHEMICAL='chemical'
    ELECTRIC='electric',
    NUCLEAR='nuclear'
    SOLAR_SAIL='solar_sail'

class Position(BaseModel):
    x:float
    y:float
    z:float

class AtlasStatus(BaseModel):
    position:Position
    velocity:float
    dist_from_earth_au:float
    dist_from_sun_au:float

class MissionRequest(BaseModel):
    launch_date:datetime
    propulsion_type:PropulsionType
    spacecraft_mass_kg:float
    fuel_budget_kg:float

class MissionResult(BaseModel):
    success:bool
    delta_v_required:float
    travel_time:int
    fuel_efficiency_Score:float=Field(ge=1, le=10)
    mission_cost_millions:float
    intercept_distance_km:float
    trajectory: List[Dict[str, float]]
    message:str
    