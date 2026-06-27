from collections import deque


class IMUProcessor:
    """Simple IMU smoothing and stabilization helper."""

    def __init__(self, window_size=10):
        self.window_size = window_size
        self.gyro_history = deque(maxlen=window_size)
        self.accel_history = deque(maxlen=window_size)

    def add_sample(self, imu_data):
        gyro = imu_data.get('gyro', {})
        accel = imu_data.get('accel', {})
        self.gyro_history.append(gyro)
        self.accel_history.append(accel)

    def get_smoothed(self):
        def average(history):
            if not history:
                return {}
            keys = history[0].keys()
            out = {}
            for key in keys:
                out[key] = sum(item.get(key, 0.0) for item in history) / len(history)
            return out

        return {
            'gyro': average(self.gyro_history),
            'accel': average(self.accel_history)
        }

    def get_attitude_adjustment(self):
        smooth = self.get_smoothed()
        gyro = smooth.get('gyro', {})
        accel = smooth.get('accel', {})
        roll = accel.get('roll', 0.0)
        pitch = accel.get('pitch', 0.0)
        yaw = gyro.get('yaw', 0.0)

        return {
            'roll_adjust': -roll * 0.1,
            'pitch_adjust': -pitch * 0.1,
            'yaw_adjust': -yaw * 0.05
        }

    @staticmethod
    def parse_payload(payload):
        if isinstance(payload, str):
            import json
            payload = json.loads(payload)
        return {
            'gyro': payload.get('gyro', {}),
            'accel': payload.get('accel', {}),
            'mag': payload.get('mag', {}),
            'temperature': payload.get('temperature')
        }
