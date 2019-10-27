class PID(object):
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_intergral = 0
        self.error_prev = None

    def control(self, error):
        self.error_intergral += error
        if self.error_prev is None:
            self.error_prev = error
        error_derivative = error - self.error_prev
        self.error_prev = error
        return self.kp*error + self.ki*self.error_intergral + self.kd*error_derivative

    def reset(self):
        self.error_prev = None
        self.error_intergral = 0