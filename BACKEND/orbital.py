import math 
from datetime import datetime,timedelta
from models import Position

class AtlasTracker:
    def __init__(self):
        self.discovery_date=datetime(2019,12,28)
        self.perihelion_distance=0.295
        self.speed_at_perihelion=41.5


    def get_current_position(self)->Position:
        return self.get_position_at_date(datetime.now())
    

    def get_position_at_date(self,target_date:datetime)->Position:
        days_since_discovery=(target_date-self.discovery_date).total_seconds()/86400

        t=days_since_discovery/365.25

        x=0.3+15*t
        y=2 + 8 * t * math.sin(t * 0.3)
        z=1 + 4 * t * math.cos(t * 0.2)

        return Position(x=x,y=y,z=z)
    
    def calculate_speed(self,target_date:datetime)->float:
        days_since_discovery = (target_date - self.discovery_date).total_seconds() / 86400
        base_speed = 41.5
        distance_factor = max(0.1, 1 / (1 + days_since_discovery / 365))

        return base_speed * distance_factor
    
    def distance_from_earth(self, atlas_pos: Position, date: datetime)->float:
        earth_angle = (date.timetuple().tm_yday / 365.25) * 2 * math.pi
        earth_x = math.cos(earth_angle)
        earth_y = math.sin(earth_angle)
        earth_z = 0

        dx = atlas_pos.x - earth_x
        dy = atlas_pos.y - earth_y  
        dz = atlas_pos.z - earth_z

        return math.sqrt(dx*dx + dy*dy + dz*dz)



