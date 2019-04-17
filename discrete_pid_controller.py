from utils import clamp


class DiscretePIDController:
    """
    Implements a discrete PID controller.
    The integrative was discretized using the Tustin transform, while the derivative was discretized using the
    Backward Euler transform.
    """
    def __init__(self, kp, ki, kd, max_command, sample_time):
        """
        Creates a discrete PID controller.

        :param kp: proportional gain.
        :type kp: float.
        :param ki: integrative gain.
        :type ki: float.
        :param kd: derivative gain.
        :type kd: float.
        :param max_command: maximum allowed command (for anti-windup).
        :type max_command: float.
        :param sample_time: sample time of the controller.
        :type sample_time: float.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        # Computing the elements of the transfer function in Z-domain
        self.a0 = kp + ki * sample_time / 2.0 + kd / sample_time
        self.a1 = -kp + ki * sample_time / 2.0 - 2.0 * kd / sample_time
        self.a2 = kd / sample_time
        self.max_command = max_command
        self.sample_time = sample_time
        self.up = 0.0  # u[k-1] (previous controller output)
        self.ep = 0.0  # e[k-1] (previous controller error)
        self.epp = 0.0  # e[k-2] (controller error two time steps before)

    def reset(self):
        """
        Resets the controller.
        """
        # Resetting filter's states
        self.up = 0.0
        self.ep = 0.0
        self.epp = 0.0

    def set_gains(self, kp, ki, kd):
        """
        Sets new gains.

        :param kp: new proportional gain.
        :type kp: float.
        :param ki: new integrative gain.
        :type ki: float.
        :param kd: new derivative gain.
        :type kd: float.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        # Recomputing the elements of the transfer function in Z-domain
        self.a0 = kp + ki * self.sample_time / 2.0 + kd / self.sample_time
        self.a1 = -kp + ki * self.sample_time / 2.0 - 2.0 * kd / self.sample_time
        self.a2 = kd / self.sample_time
        self.reset()

    def control(self, error):
        """
        Updates the controller.

        :param error: current error.
        :type error: float.
        :return: the command which should be sent to the actuator.
        :rtype: float.
        """
        u = self.up + self.a0 * error + self.a1 * self.ep + self.a2 * self.epp
        # Applying anti-windup by clamping the command
        u = clamp(u, -self.max_command, self.max_command)
        self.epp = self.ep
        self.ep = error
        self.up = u
        return u
