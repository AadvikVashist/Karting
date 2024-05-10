import numpy as np
from path_manipulation import calculate_curvature
class Kart:
    def __init__(self, max_speed, acceleration, braking, turning_radius, steerability, width=1.0, length=1.0):
        self.max_speed = max_speed
        self.acceleration = acceleration
        self.braking = braking
        self.turning_radius = turning_radius
        self.width = width
        self.length = length
        self.max_angular_acceleration = steerability

    def get_max_speed(self, curvature):
        # Constants based on kart capabilities and track conditions

        # Speed based on lateral acceleration limit (v^2/r = a)
        if curvature != 0:
            max_speed = (self.max_angular_acceleration / curvature) ** 0.5
        else:
            max_speed = self.max_speed
        
        return min(max_speed, self.max_speed)

    def get_max_acceleration(self, speed):
        # Returns the acceleration based on current speed
        return self.acceleration if speed < self.max_speed else 0

    def get_braking_distance(self, speed):
        # Estimate the stopping distance based on speed and braking capacity
        stopping_distance = speed ** 2 / (2 * self.braking)
        return stopping_distance
    def get_curvature(self, points):
        return calculate_curvature(points)