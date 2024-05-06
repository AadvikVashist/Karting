class Kart:
    def __init__(self, max_speed, acceleration, braking, turning_radius, steerability, width=1.0, length=1.0):
        self.max_speed = max_speed
        self.acceleration = acceleration
        self.braking = braking
        self.turning_radius = turning_radius
        self.width = width
        self.length = length
        self.max_angular_acceleration = steerability
