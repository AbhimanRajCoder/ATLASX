import math
from datetime import datetime, timezone
from typing import List, Dict, Union
from pydantic import BaseModel
from orbital import AtlasTracker
from models import MissionRequest, PropulsionType


class MissionResult(BaseModel):
    success: bool
    delta_v_required: float
    travel_time: int  # in days
    fuel_efficiency_Score: int
    mission_cost_millions: float
    intercept_distance_km: float
    message: str
    trajectory: List[Dict[str, float]]  # time in days, distance in AU, velocity in km/s


class MissionPlanner:
    def __init__(self):
        self.atlas_tracker = AtlasTracker()

        # Sun's gravitational parameter (km^3/s^2)
        # More accurate value
        self.mu_sun = 1.32712440018e11

        # Astronomical Unit in km
        self.AU = 149597870.7

        # Seconds per day
        self.seconds_per_day = 86400.0

        # Earth's standard orbital radius (assumed circular for Hohmann) in km
        self.r_earth = 1.0 * self.AU

        self.propulsion_data = {
            PropulsionType.CHEMICAL: {
                "max_delta_v": 15.0,  # km/s
                "cost_per_kg": 50000,
                "efficiency": 0.7,
                "min_travel_time": 300,  # days
                "exhaust_velocity": 4.5,  # km/s
            },
            PropulsionType.ELECTRIC: {
                "max_delta_v": 25.0,
                "cost_per_kg": 80000,
                "efficiency": 0.9,
                "min_travel_time": 800,  # days
                "exhaust_velocity": 30.0,  # km/s
            },
            PropulsionType.NUCLEAR: {
                "max_delta_v": 35.0,
                "cost_per_kg": 200000,
                "efficiency": 0.85,
                "min_travel_time": 250,  # days
                "exhaust_velocity": 9.0,  # km/s
            },
            PropulsionType.SOLAR_SAIL: {
                "max_delta_v": 10.0,
                "cost_per_kg": 30000,
                "efficiency": 0.95,
                "min_travel_time": 1200,  # days
                "exhaust_velocity": float("inf"),  # no propellant consumption
            },
        }

    def _normalize_distance_km(self, raw: Union[float, object]) -> float:
        """
        Normalize AtlasTracker distance or position object into km.
        This attempts to detect whether the returned value is in AU or km.
        """
        if raw is None:
            raise ValueError("AtlasTracker returned no distance")

        # If it looks like a vector/position object
        if hasattr(raw, "x") and hasattr(raw, "y"):
            z_val = getattr(raw, "z", 0.0)
            dist_val = math.sqrt(raw.x**2 + raw.y**2 + z_val**2)
        elif hasattr(raw, "distance"):
            dist_val = float(raw.distance)
        elif hasattr(raw, "radius"):
            dist_val = float(raw.radius)
        else:
            dist_val = float(raw)

        # Heuristics:
        # - If the number is small (<10000) it's probably in AU (e.g., 0.1 - 30)
        # - If it is large (>1e6) treat as km
        if 0.001 <= dist_val <= 1e4:
            # assume AU
            return dist_val * self.AU
        elif dist_val > 1e6:
            return dist_val
        else:
            # fallback: assume AU
            return dist_val * self.AU

    # -------------------------
    # Orbital mechanics helpers
    # -------------------------

    def _hohmann_delta_v(self, r1_km: float, r2_km: float) -> float:
        """
        Compute two-impulse Hohmann transfer delta-v (km/s) between circular orbits r1 and r2 about the Sun.
        Returns the sum of both impulses (transfer injection + circularization).
        """
        mu = self.mu_sun
        r1 = r1_km
        r2 = r2_km

        # Circular speeds
        v1 = math.sqrt(mu / r1)
        v2 = math.sqrt(mu / r2)

        # Semi-major axis of transfer ellipse
        a_t = 0.5 * (r1 + r2)

        # Speeds on transfer ellipse at periapsis (r1) and apoapsis (r2)
        v_peri = math.sqrt(mu * (2.0 / r1 - 1.0 / a_t))
        v_apo = math.sqrt(mu * (2.0 / r2 - 1.0 / a_t))

        # Delta-vs for injection and circularization
        delta_v1 = abs(v_peri - v1)
        delta_v2 = abs(v2 - v_apo)

        return delta_v1 + delta_v2

    def _hohmann_transfer_time_days(self, r1_km: float, r2_km: float) -> float:
        """
        Hohmann transfer time = half the period of the transfer ellipse.
        Returns time in days.
        """
        mu = self.mu_sun
        a_t = 0.5 * (r1_km + r2_km)
        # Period (s): T = 2*pi*sqrt(a^3/mu) ; transfer time = T/2
        T = 2.0 * math.pi * math.sqrt(a_t ** 3 / mu)
        transfer_time_seconds = T / 2.0
        return transfer_time_seconds / self.seconds_per_day

    def _kepler_solve_E(self, M: float, e: float, tol: float = 1e-8, max_iter: int = 100) -> float:
        """
        Solve Kepler's equation M = E - e*sin(E) for E (eccentric anomaly) using Newton-Raphson.
        M must be in radians, result E in radians.
        """
        if e < 0.8:
            E = M
        else:
            E = math.pi

        for _ in range(max_iter):
            f = E - e * math.sin(E) - M
            fprime = 1 - e * math.cos(E)
            if abs(fprime) < 1e-12:
                break
            dE = -f / fprime
            E += dE
            if abs(dE) < tol:
                break
        return E

    def _generate_transfer_trajectory(self, r1_km: float, r2_km: float, num_points: int = 40) -> List[Dict[str, float]]:
        """
        Build a set of trajectory points along a Hohmann transfer ellipse between r1 and r2 (heliocentric radii).
        Each point contains:
            - time (days since launch)
            - distance (AU from Sun)
            - velocity (km/s, heliocentric instantaneous speed)
        Uses full Kepler propagation for the transfer (solving Kepler's equation).
        """
        points: List[Dict[str, float]] = []

        # Transfer semi-major axis and eccentricity (ellipse)
        a_t = 0.5 * (r1_km + r2_km)
        # Eccentricity from r_peri = a(1-e) = r1 -> e = 1 - r1/a
        e = 1.0 - (r1_km / a_t)
        # Mean motion n = sqrt(mu/a^3)
        n = math.sqrt(self.mu_sun / (a_t ** 3))

        # Transfer time in seconds (half period)
        transfer_time_seconds = math.pi * math.sqrt(a_t ** 3 / self.mu_sun)
        # We'll sample times evenly from 0 -> transfer_time_seconds
        for i in range(num_points + 1):
            t_frac = i / num_points
            t_sec = t_frac * transfer_time_seconds
            # Mean anomaly for the transfer ellipse (start at periapsis M=0)
            M = n * t_sec
            # Solve for eccentric anomaly E
            E = self._kepler_solve_E(M, e)
            # Radius at this E: r = a * (1 - e*cos E)
            r_km = a_t * (1.0 - e * math.cos(E))
            # True anomaly (theta)
            # theta = 2 * atan2( sqrt(1+e) * sin(E/2), sqrt(1-e) * cos(E/2) )
            sin_half = math.sin(E / 2.0)
            cos_half = math.cos(E / 2.0)
            denom = math.sqrt(max(1e-12, (1.0 - e))) * cos_half
            numer = math.sqrt(1.0 + e) * sin_half
            theta = 2.0 * math.atan2(numer, denom)

            # Instantaneous speed from vis-viva
            v_kms = math.sqrt(self.mu_sun * (2.0 / r_km - 1.0 / a_t))

            points.append({
                "time": t_sec / self.seconds_per_day,  # days
                "distance": r_km / self.AU,  # AU
                "velocity": v_kms,  # km/s
                "true_anomaly_rad": theta,
            })

        return points

    # -------------------------
    # Fixed Calculation helpers
    # -------------------------

    def _calculate_required_delta_v(self, distance_from_earth_km: float, propulsion_type: PropulsionType,
                                    spacecraft_mass_kg: float) -> float:
        """
        Use Hohmann-transfer-based estimate for delta-v from Earth circular orbit (1 AU) to target orbit.
        We need an estimate for target heliocentric radius:
          - We only have distance_from_earth_km (Earth-to-asteroid separation).
          - As a reasonable approximation, assume the asteroid is 'outward' from Earth:
            r_target = 1 AU + distance_from_earth (in AU). Cap to reasonable bounds.
        This is an approximation — a full ephemeris-based heliocentric vector is needed for exact results.
        """
        # Convert to AU
        distance_au = distance_from_earth_km / self.AU

        # Approximate heliocentric radius of target (assume outward direction)
        r_target_au = 1.0 + distance_au
        # enforce sensible bounds: not less than 0.2 AU, not greater than, say, 50 AU
        r_target_au = max(0.2, min(50.0, r_target_au))

        r1_km = self.r_earth
        r2_km = r_target_au * self.AU

        # Hohmann delta-v (km/s)
        hohmann_dv = self._hohmann_delta_v(r1_km, r2_km)

        # Add small rendezvous budget (matching asteroid velocity, small correction)
        rendezvous_budget = 0.5  # km/s (order-of-magnitude)
        # Add LEO-to-escape budget (~3.2 km/s commonly used)
        escape_budget = 3.2

        total_delta_v = escape_budget + hohmann_dv + rendezvous_budget

        # Adjust for propulsion efficiency (non-ideal)
        prop_specs = self.propulsion_data[propulsion_type]
        effective_delta_v = total_delta_v / prop_specs["efficiency"]

        return effective_delta_v

    def _calculate_travel_time(self, distance_from_earth_km: float, min_time_days: int,
                               propulsion_type: PropulsionType) -> int:
        """
        Use Hohmann transfer time as base, then enforce min_time_days and propulsion modifiers.
        """
        distance_au = distance_from_earth_km / self.AU
        r_target_au = 1.0 + distance_au
        r_target_au = max(0.2, min(50.0, r_target_au))

        r1_km = self.r_earth
        r2_km = r_target_au * self.AU

        hohmann_days = int(math.ceil(self._hohmann_transfer_time_days(r1_km, r2_km)))

        # Enforce minimum and propulsion limits
        travel_time_days = max(min_time_days, hohmann_days)

        if propulsion_type == PropulsionType.NUCLEAR:
            travel_time_days = int(travel_time_days * 0.8)
        elif propulsion_type == PropulsionType.ELECTRIC:
            # Electric propulsion often trades time for delta-v capability; we'll allow slightly longer cruise
            travel_time_days = int(travel_time_days * 1.2)
        elif propulsion_type == PropulsionType.SOLAR_SAIL:
            travel_time_days = int(travel_time_days * 1.8)

        return int(travel_time_days)

    def _calculate_fuel_efficiency(self, delta_v: float, fuel_budget: float, spacecraft_mass: float,
                                   prop_specs: dict) -> int:
        if prop_specs["exhaust_velocity"] == float("inf"):
            return 10  # Solar sail: best-case

        # Tsiolkovsky rocket equation: m0 = m1 * exp(delta_v / v_e)
        try:
            required_initial_mass = spacecraft_mass * math.exp(delta_v / prop_specs["exhaust_velocity"])
        except OverflowError:
            return 1
        required_fuel = required_initial_mass - spacecraft_mass

        if required_fuel <= 0.0:
            return 10

        if fuel_budget >= required_fuel:
            fuel_ratio = fuel_budget / required_fuel
            efficiency_score = min(10, max(1, int(1 + (fuel_ratio - 1) * 4)))  # scale sensibly
        else:
            efficiency_score = max(1, int(10 * (fuel_budget / required_fuel)))

        return efficiency_score

    def _calculate_cost(self, mass: float, fuel: float, cost_per_kg: float, propulsion_type: PropulsionType) -> float:
        spacecraft_cost = mass * cost_per_kg / 1e6
        if propulsion_type == PropulsionType.SOLAR_SAIL:
            fuel_cost = 0.0
        else:
            fuel_cost = fuel * (cost_per_kg * 0.2) / 1e6
        operations_cost = 50.0 + (mass / 1000.0) * 10.0
        return spacecraft_cost + fuel_cost + operations_cost

    def _calculate_intercept_distance(self, fuel_efficiency: int, is_possible: bool) -> float:
        """
        Returns planned rendezvous offset from asteroid center (km). This is a mission planning parameter,
        not the heliocentric transfer endpoint. Small numbers mean closer rendezvous/landing capability.
        """
        if is_possible:
            if fuel_efficiency >= 8:
                return 1000.0  # km
            elif fuel_efficiency >= 5:
                return 5000.0
            else:
                return 15000.0
        else:
            return 100000.0  # flyby

    def _generate_trajectory_points(self, distance_from_earth_km: float, intercept_offset_km: float,
                                    travel_time_days: int, success: bool, propulsion_type: PropulsionType) -> List[Dict[str, float]]:
        """
        Generate a realistic heliocentric transfer trajectory from Earth's orbit (1 AU) to the target
        heliocentric radius using Kepler propagation on the transfer ellipse. Output includes:
            - time (days since launch)
            - distance (AU from Sun)
            - velocity (km/s heliocentric)
        The end of this transfer corresponds to heliocentric rendezvous radius; the relative intercept offset
        (intercept_offset_km) is the final approach offset to the asteroid (local rendezvous distance).
        """
        # Approximate target heliocentric radius as 1 AU + separation (assume outward)
        distance_au = distance_from_earth_km / self.AU
        r_target_au = 1.0 + distance_au
        r_target_au = max(0.2, min(50.0, r_target_au))

        r1_km = self.r_earth
        r2_km = r_target_au * self.AU

        # Build transfer trajectory (heliocentric)
        transfer_points = self._generate_transfer_trajectory(r1_km, r2_km, num_points=40)

        # Now convert to required output format:
        # time in days (int), distance in AU, velocity in km/s. Smooth any unrealistic spikes.
        traj_out: List[Dict[str, float]] = []
        last_v = None
        for p in transfer_points:
            t_days = p["time"]
            dist_au = p["distance"]
            v_kms = p["velocity"]

            # Smooth velocity changes by simple low-pass (moving average style)
            if last_v is None:
                smoothed_v = v_kms
            else:
                smoothed_v = 0.3 * v_kms + 0.7 * last_v
            last_v = smoothed_v

            traj_out.append({
                "time": round(t_days, 3),
                "distance": round(dist_au, 8),
                "velocity": round(smoothed_v, 6),
            })

        # Append final approach phase (local asteroid approach) as a few points reducing relative distance
        # from rendezvous offset down to the intercept_offset_km (we don't attempt a landed state).
        final_time = transfer_points[-1]["time"] / self.seconds_per_day if False else transfer_points[-1]["time"]
        # Using days already in transfer_points
        base_day = transfer_points[-1]["time"]  # days

        approach_steps = 5
        # Current heliocentric distance in AU at rendezvous
        rendezvous_au = r2_km / self.AU
        # Use local relative velocities small compared to heliocentric speeds (0.1 - 1 km/s)
        for j in range(1, approach_steps + 1):
            frac = j / approach_steps
            # time after arrival in days
            t_after = base_day + frac * min(10.0, travel_time_days * 0.01)
            # distance remains heliocentric rendezvous radius, but we keep a note of intercept offset as local approach
            traj_out.append({
                "time": round(t_after, 3),
                "distance": round(rendezvous_au, 8),
                "velocity": round(max(0.01, last_v * (1.0 - 0.1 * frac)), 6),
            })

        return traj_out

    def _generate_message(self, success: bool, delta_v: float, prop_specs: dict) -> str:
        if not success:
            return f"Mission impossible: Requires {delta_v:.1f} km/s but {prop_specs['max_delta_v']:.1f} km/s is maximum for this propulsion"
        if delta_v < prop_specs["max_delta_v"] * 0.6:
            return f"Excellent mission! Only {delta_v:.1f} km/s required - plenty of fuel margin"
        elif delta_v < prop_specs["max_delta_v"] * 0.8:
            return f"Good mission! {delta_v:.1f} km/s required - moderate fuel usage"
        else:
            return f"Challenging mission! {delta_v:.1f} km/s required - high fuel consumption"

    # -------------------------
    # Main planner
    # -------------------------

    def plan_mission(self, request: MissionRequest) -> MissionResult:
        # Parse launch_date
        if isinstance(request.launch_date, datetime):
            launch_date = request.launch_date
        else:
            launch_date = datetime.fromisoformat(request.launch_date)

        if launch_date.tzinfo is None:
            launch_date = launch_date.replace(tzinfo=timezone.utc)
        else:
            launch_date = launch_date.astimezone(timezone.utc)

        tracker_date = launch_date.replace(tzinfo=None)

        # Get asteroid position at launch (distance from Earth)
        atlas_pos = self.atlas_tracker.get_position_at_date(tracker_date)
        raw_distance = self.atlas_tracker.distance_from_earth(atlas_pos, tracker_date)

        # Normalize distance to km (distance_from_earth)
        distance_at_launch_km = self._normalize_distance_km(raw_distance)

        # Calculate Δv using Hohmann-based estimate
        total_delta_v = self._calculate_required_delta_v(
            distance_at_launch_km,
            request.propulsion_type,
            request.spacecraft_mass_kg,
        )

        prop_specs = self.propulsion_data[request.propulsion_type]
        is_possible = total_delta_v <= prop_specs["max_delta_v"]

        # Travel time (days) based on Hohmann + propulsion
        travel_time_days = self._calculate_travel_time(
            distance_at_launch_km,
            prop_specs["min_travel_time"],
            request.propulsion_type,
        )

        # Fuel efficiency (score 1-10)
        fuel_efficiency = self._calculate_fuel_efficiency(
            total_delta_v,
            request.fuel_budget_kg,
            request.spacecraft_mass_kg,
            prop_specs,
        )

        # Cost estimate (millions)
        mission_cost = self._calculate_cost(
            request.spacecraft_mass_kg,
            request.fuel_budget_kg,
            prop_specs["cost_per_kg"],
            request.propulsion_type,
        )

        # Rendezvous/intercept offset (local approach distance, km)
        intercept_distance_km = self._calculate_intercept_distance(
            fuel_efficiency,
            is_possible,
        )

        # Trajectory points (heliocentric transfer + approach)
        trajectory_points = self._generate_trajectory_points(
            distance_at_launch_km,
            intercept_distance_km,
            travel_time_days,
            is_possible,
            request.propulsion_type,
        )

        message = self._generate_message(is_possible, total_delta_v, prop_specs)

        return MissionResult(
            success=is_possible,
            delta_v_required=round(total_delta_v, 3),
            travel_time=travel_time_days,
            fuel_efficiency_Score=fuel_efficiency,
            mission_cost_millions=round(mission_cost, 2),
            intercept_distance_km=intercept_distance_km,
            trajectory=trajectory_points,
            message=message,
        )
