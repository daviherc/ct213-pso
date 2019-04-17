class LowPassFilter:
    """
    Represents a first-order digital low pass filter. The transfer function was discretized using the Tustin transform.
    """
    def __init__(self, bandwidth, sample_time):
        """
        Creates the low pass filter.

        :param bandwidth: filter's bandwidth in radians.
        :type bandwidth: float.
        :param sample_time: sample time of the filter.
        :type sample_time: float.
        """
        self.bandwidth = bandwidth
        self.sample_time = sample_time
        denominator = bandwidth * sample_time + 2.0
        # Computing the elements of the transfer function in Z-domain
        self.b1 = (bandwidth * sample_time - 2.0) / denominator
        self.a0 = (bandwidth * sample_time) / denominator
        self.a1 = (bandwidth * sample_time) / denominator
        self.up = 0.0  # u[k-1] (previous filter input)
        self.yp = 0.0  # y[k-1] (previous filter output)

    def reset(self):
        """
        Resets the filter.
        """
        self.up = 0.0
        self.yp = 0.0

    def filter(self, input_value):
        y = -self.b1 * self.up + self.a0 * input_value + self.a1 * self.yp
        self.up = input_value
        self.yp = y
        return y
