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
        # Calculate the maximum speed based on the curvature of the racing line
        return min(self.max_speed, np.sqrt(self.turning_radius / curvature))

    def get_max_acceleration(self, speed):
        # Returns the acceleration based on current speed
        return self.acceleration if speed < self.max_speed else 0

    def get_braking_distance(self, speed):
        # Estimate the stopping distance based on speed and braking capacity
        stopping_distance = speed ** 2 / (2 * self.braking)
        return stopping_distance
    def get_curvature(self, points):
        return calculate_curvature(points)