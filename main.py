
from environment.vehicle_env import VehicleEnvironment

if __name__ == "__main__":
    vehicle_env = VehicleEnvironment()
    num_sessions = 3
    for session in range(num_sessions):
        vehicle_env.update_session(session)
