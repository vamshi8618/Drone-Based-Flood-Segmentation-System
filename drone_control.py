import json
import requests


class DroneController:
    """WiFi-based interface for NodeMCU / ESP8266 drone control."""

    def __init__(self, host='192.168.4.1', port=80, timeout=2.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.base_url = f'http://{self.host}:{self.port}'

    def send_command(self, command, params=None):
        payload = {
            'command': command,
            'params': params or {}
        }
        url = f'{self.base_url}/command'
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def takeoff(self):
        return self.send_command('takeoff')

    def land(self):
        return self.send_command('land')

    def hover(self):
        return self.send_command('hover')

    def emergency_stop(self):
        return self.send_command('stop')

    def set_velocity(self, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0):
        params = {
            'vx': float(vx),
            'vy': float(vy),
            'vz': float(vz),
            'yaw_rate': float(yaw_rate)
        }
        return self.send_command('velocity', params=params)

    def follow_distance(self, distance_meters=5.0):
        return self.send_command('follow_distance', params={'distance': float(distance_meters)})

    def set_attitude(self, roll=0.0, pitch=0.0, yaw=0.0, throttle=0.0):
        params = {
            'roll': float(roll),
            'pitch': float(pitch),
            'yaw': float(yaw),
            'throttle': float(throttle)
        }
        return self.send_command('attitude', params=params)

    def get_telemetry(self):
        url = f'{self.base_url}/telemetry'
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()


class DroneTelemetry:
    @staticmethod
    def parse(raw_data):
        if isinstance(raw_data, str):
            raw_data = json.loads(raw_data)

        return {
            'battery': raw_data.get('battery'),
            'altitude': raw_data.get('altitude'),
            'gps': raw_data.get('gps'),
            'imu': raw_data.get('imu'),
            'state': raw_data.get('state')
        }
