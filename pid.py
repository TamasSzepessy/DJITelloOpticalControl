class PID(object):
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_int = 0
        self.error_prev = None

    def control(self, error):
        self.error_int += error
        if self.error_prev is None:
            self.error_prev = error
        error_deriv = error - self.error_prev
        self.error_prev = error
        # y = self.kp*error + self.ki*self.error_int + self.kd*error_deriv
        # # limit to 100 for drone
        # if y < -100: y = -100
        # if y > 100:  y = 100
        return self.kp*error + self.ki*self.error_int + self.kd*error_deriv

    def reset(self):
        self.error_prev = None
        self.error_int = 0