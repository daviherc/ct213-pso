from math import sin, cos, fabs
from discrete_pid_controller import DiscretePIDController
from constants import ROBOT_SAMPLE_TIME, SIMULATION_SAMPLE_TIME, DETECTION_THRESHOLD
from utils import clamp, Vector2
from low_pass_filter import LowPassFilter


class LineSensorArray:
    """
    Represents a line sensor array.
    """
    def __init__(self, params):
        """
        Creates a line sensor array.
        Parameters:
            sensor_range: the measurement range in meters of each individual line sensor.
            num_sensors: the number of line sensors in the array.
            array_width: the width in meters of the array (distance from the first to the last sensor).

        :param params: line sensor array parameters.
        :type params: Params.
        """
        self.sensor_range = params.sensor_range
        self.num_sensors = params.num_sensors
        self.array_width = params.array_width
        self.sensors_positions = [0.0] * params.num_sensors
        self.intensity = [0.0] * params.num_sensors
        self.define_sensors_positions()

    def define_sensors_positions(self):
        """
        Computes the one-dimensional position of each sensor from the given parameters.
        The origin of the coordinates is the center of the array.
        """
        min_position = -self.array_width / 2
        distance_between_sensors = self.array_width / (self.num_sensors - 1)
        for i in range(self.num_sensors):
            self.sensors_positions[i] = min_position + distance_between_sensors * i

    def set_intensity(self, intensity):
        """
        Sets the measured intensities of each sensor.

        :param intensity: a list of floats containing the intensity measured by each sensor.
        :type intensity: list of floats.
        """
        for i in range(self.num_sensors):
            self.intensity[i] = intensity[i]

    def get_error(self):
        """
        Computes the line error using a center of mass algorithm.
        e = sum(y[i] * I[i]) / sum(I[i]), where y[i] is the position of sensor i and I[i] is the intensity of the
        respective sensor. The sums iterate over all i.
        Moreover, a boolean indicating if the a line was detected is also returned.

        :return error: the line error with respect to the center of the array.
        :rtype error: float.
        :return detection: if a line was detected.
        :rtype detection: boolean.
        """
        detection = False
        error = 0.0
        sum_intensity = 0.0
        for i in range(self.num_sensors):
            error += self.sensors_positions[i] * self.intensity[i]
            sum_intensity += self.intensity[i]
        if sum_intensity > DETECTION_THRESHOLD:
            error /= sum_intensity
            detection = True
        return error, detection


class LineFollower:
    """
    Represents a line follower robot.
    """
    def __init__(self, pose, controller_params, robot_params, sensor_params):
        """
        Creates a line follower robot.
        Controller parameters:
            max_linear_speed_command: the linear speed commanded to the robot.
            kp: proportional gain of the angle controller.
            ki: integrative gain of the angle controller.
            kd: derivative gain of the angle controller.
        Robot parameters:
            sensor_offset: offset in x coordinate between the wheels' axle center and the sensor array center
            max_wheel_speed: maximum wheel speed
            wheel_radius: radius of the wheel
            wheels_distance: distance between wheels
        Sensor parameters:
            sensor_range: the measurement range in meters of each individual line sensor.
            num_sensors: the number of line sensors in the array.
            array_width: the width in meters of the array (distance from the first to the last sensor).

        :param pose: the initial pose of the robot.
        :type pose: Pose.
        :param controller_params: parameters used for the angle controller.
        :type controller_params: Params.
        :param robot_params: parameters used for the robot body.
        :type robot_params: Params.

        """
        self.pose = pose
        # These variables are used to define reference speeds which will be fed to the wheels' dynamics
        self.reference_linear_speed = 0.0
        self.reference_angular_speed = 0.0
        # Since the robot control may have delays, we have to add issued commands to a buffer
        self.linear_speed_commands = []
        self.angular_speed_commands = []
        # Robot control delay
        self.delay = 2
        self.max_linear_speed_command = controller_params.max_linear_speed_command
        # Collecting robot parameters
        self.sensor_offset = robot_params.sensor_offset
        self.max_wheel_speed = robot_params.max_wheel_speed
        self.wheel_radius = robot_params.wheel_radius
        self.wheels_distance = robot_params.wheels_distance
        # Since the maximum speed is actually saturated at the wheels, we compute the maximum
        # allowed angular speed for the desired linear speed
        max_wheel_linear = self.max_linear_speed_command / self.wheel_radius
        max_wheel_angular = clamp(self.max_wheel_speed - max_wheel_linear, 0.0, self.max_wheel_speed)
        # Creating the angle controller
        self.controller = DiscretePIDController(controller_params.kp, controller_params.ki, controller_params.kd,
                                                2.0 * max_wheel_angular * self.wheel_radius / self.wheels_distance,
                                                ROBOT_SAMPLE_TIME)
        # Creating the line sensor
        self.line_sensor = LineSensorArray(sensor_params)
        # Low pass filters are used to simulate the wheels' dynamics
        self.left_wheel_dynamics = LowPassFilter(robot_params.wheel_bandwidth, SIMULATION_SAMPLE_TIME)
        self.right_wheel_dynamics = LowPassFilter(robot_params.wheel_bandwidth, SIMULATION_SAMPLE_TIME)
        # In order to simulate the fact that the robot control frequency may be slower than the simulation frequency,
        # we define a control frequency divider.
        self.control_frequency_divider = round(ROBOT_SAMPLE_TIME / SIMULATION_SAMPLE_TIME)
        self.iteration = 0

    def reset(self, pose, controller_params=None):
        """
        Resets the line follower robot.
        Changing controller parameters is optional. If no controller parameters is passed when calling this
        method, the previous controller parameters will be maintained.

        :param pose: the pose of the robot after the reset.
        :type pose: Pose.
        :param controller_params: new controller parameters.
        :type controller_params: Params.
        """
        self.pose = pose
        # If new controller parameters where passed, we also update the controller parameters.
        if controller_params is not None:
            self.max_linear_speed_command = controller_params.max_linear_speed_command
            self.controller.set_gains(controller_params.kp, controller_params.ki, controller_params.kd)
        # To guarantee that the robot behavior will be reproducible, we have to reset all variables which
        # influence its dynamics
        self.left_wheel_dynamics.reset()
        self.right_wheel_dynamics.reset()
        self.controller.reset()
        self.linear_speed_commands = []
        self.angular_speed_commands = []
        self.reference_linear_speed = 0.0
        self.reference_angular_speed = 0.0
        self.iteration = 0

    def unicycle_to_wheels(self, linear_speed, angular_speed):
        """
        Converts from speeds of the unicycle model to wheels' speeds

        :param linear_speed: linear speed.
        :type linear_speed: float.
        :param angular_speed: angular speed.
        :type angular_speed: float.
        :return right_speed: speed of the right wheel.
        :rtype right_speed: float.
        :return left_speed: speed of the left wheel.
        :rtype left_speed: float.
        """
        right_speed = (1.0 / self.wheel_radius) * (linear_speed + angular_speed * self.wheels_distance / 2.0)
        left_speed = (1.0 / self.wheel_radius) * (linear_speed - angular_speed * self.wheels_distance / 2.0)
        return right_speed, left_speed

    def wheels_to_unicycle(self, right_speed, left_speed):
        """
        Converts from wheels' speeds of the unicycle model.

        :param right_speed: speed of the right wheel.
        :type right_speed: float.
        :param left_speed: speed of the left wheel.
        :type left_speed: float.
        :return linear_speed: linear speed.
        :rtype linear_speed: float.
        :return angular_speed: angular speed.
        :rtype angular_speed: float.
        """
        linear_speed = (right_speed + left_speed) * self.wheel_radius / 2.0
        angular_speed = (right_speed - left_speed) * self.wheel_radius / self.wheels_distance
        return linear_speed, angular_speed

    def get_sensors_global_positions(self):
        """
        Obtains the positions of the sensors in the global coordinate system.

        :return: global positions of the sensors.
        :rtype: list of Vector2.
        """
        sensor_center = Vector2(self.pose.position.x, self.pose.position.y)
        sensor_center.x += self.sensor_offset * cos(self.pose.rotation)
        sensor_center.y += self.sensor_offset * sin(self.pose.rotation)
        global_positions = []
        for i in range(self.line_sensor.num_sensors):
            position = Vector2(sensor_center.x, sensor_center.y)
            position.x += -self.line_sensor.sensors_positions[i] * sin(self.pose.rotation)
            position.y += self.line_sensor.sensors_positions[i] * cos(self.pose.rotation)
            global_positions.append(position)
        return global_positions

    def set_line_sensor_intensity(self, intensity):
        """
        Sets the intensity of the line sensor array.

        :param intensity: intensities measured by each line sensor.
        :type intensity: list of floats.
        """
        self.line_sensor.set_intensity(intensity)

    def get_velocity(self):
        """
        Obtains the unicycle velocity of the robot.

        :return: tuple containing the linear and angular speeds of the robot.
        :rtype: two-dimensional tuple of floats.
        """
        right_speed = self.right_wheel_dynamics.yp
        left_speed = self.left_wheel_dynamics.yp
        return self.wheels_to_unicycle(right_speed, left_speed)

    def set_velocity(self, linear_speed, angular_speed):
        """
        Registers a robot velocity command. Since the actuation system is delayed, the command may not be
        immediately executed.

        :param linear_speed: the robot's linear speed.
        :type linear_speed: float
        :param angular_speed: the robot's angular speed.
        :type angular_speed: float
        """
        right_speed, left_speed = self.unicycle_to_wheels(linear_speed, angular_speed)
        right_speed = clamp(right_speed, -self.max_wheel_speed, self.max_wheel_speed)
        left_speed = clamp(left_speed, -self.max_wheel_speed, self.max_wheel_speed)
        linear, angular = self.wheels_to_unicycle(right_speed, left_speed)
        if len(self.linear_speed_commands) >= self.delay:
            self.reference_linear_speed = self.linear_speed_commands[-self.delay]
        if len(self.angular_speed_commands) >= self.delay:
            self.reference_angular_speed = self.angular_speed_commands[-self.delay]
        self.linear_speed_commands.append(linear)
        self.angular_speed_commands.append(angular)
        if len(self.linear_speed_commands) > self.delay:
            self.linear_speed_commands.pop(0)
        if len(self.angular_speed_commands) > self.delay:
            self.angular_speed_commands.pop(0)

    def control(self):
        """
        Updates the robot controller.
        """
        error, detected = self.line_sensor.get_error()
        angular_speed = self.controller.control(error)
        self.set_velocity(self.max_linear_speed_command, angular_speed)

    def move(self):
        """
        Moves the robot during one time step.
        """
        dt = SIMULATION_SAMPLE_TIME
        right_command, left_command = self.unicycle_to_wheels(self.reference_linear_speed, self.reference_angular_speed)
        right_speed = self.right_wheel_dynamics.filter(right_command)
        left_speed = self.left_wheel_dynamics.filter(left_command)
        v, w = self.wheels_to_unicycle(right_speed, left_speed)
        # If the angular speed is too low, the complete movement equation fails due to a division by zero.
        # Therefore, in this case, we use the equation we arrive if we take the limit when the angular speed
        # is close to zero.
        if fabs(w) < 1.0e-3:
            self.pose.position.x += v * dt * cos(self.pose.rotation + w * dt / 2.0)
            self.pose.position.y += v * dt * sin(self.pose.rotation + w * dt / 2.0)
        else:
            self.pose.position.x += (2.0 * v / w) * cos(self.pose.rotation + w * dt / 2.0) * sin(w * dt / 2.0)
            self.pose.position.y += (2.0 * v / w) * sin(self.pose.rotation + w * dt / 2.0) * sin(w * dt / 2.0)
        self.pose.rotation += w * dt

    def update(self):
        """
        Updates the robot, including its controller.
        """
        # Since the controller update frequency time is slower than the simulation frequency,
        # we update the controller only every control_frequency_divider cycles.
        if self.iteration % self.control_frequency_divider == 0:
            self.control()
            # To avoid overflow, we reset the iteration counter
            self.iteration = self.iteration % self.control_frequency_divider
        self.move()
        self.iteration += 1
