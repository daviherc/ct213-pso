import pygame
from math import sin, cos, fabs
from utils import Pose, DrawingUtils, Params, m2pix, Vector2


class RobotSprite:
    """
    Represents the robot sprite which is used to draw the robot on screen.
    Sprite parameters:
        wheel_thickness: the thickness of the wheel.
        sensor_thickness: the thickness of the line sensor array.
        wheel_radius: the radius of the robot's wheel.
        wheels_distance: the distance between wheels.
        sensor_offset: offset in x coordinate between the wheels' axle center and the sensor array center
        array_width: width of the line sensor array.
    """
    def __init__(self, sprite_params):
        wt = sprite_params.wheel_thickness
        st = sprite_params.sensor_thickness
        r = sprite_params.wheel_radius
        d = sprite_params.wheels_distance
        o = sprite_params.sensor_offset
        sw = sprite_params.array_width
        self.body = DrawingUtils.rectangle_to_polygon((-r, -d / 2.0 + wt / 2.0, r + o, d - wt))
        self.right_wheel = DrawingUtils.rectangle_to_polygon((-r, -d / 2.0 - wt / 2.0, 2 * r, wt))
        self.left_wheel = DrawingUtils.rectangle_to_polygon((-r, d / 2.0 - wt / 2.0, 2 * r, wt))
        self.line_sensor = DrawingUtils.rectangle_to_polygon((o - st / 2.0, -sw / 2.0, st, sw))
        self.wheel_color = (0, 0, 0)
        self.robot_color = (153, 153, 0)

    @staticmethod
    def transform(pose, polygon):
        """
        Transforms (translate and rotate) a polygon by a given pose.

        :param pose: translation and rotation of the transform.
        :type pose: Pose.
        :param polygon: the polygon which will be transformed.
        :type polygon: list of Vector2.
        :return: the polygon after each is transformed accordingly.
        :rtype: list of Vector2.
        """
        transformed_polygon = []
        for point in polygon:
            x = pose.position.x + point[0] * cos(pose.rotation) - point[1] * sin(pose.rotation)
            y = pose.position.y + point[0] * sin(pose.rotation) + point[1] * cos(pose.rotation)
            transformed_polygon.append((x, y))
        return transformed_polygon

    def draw(self, window, pose):
        """
        Draws the robot sprite on the screen.

        :param window: pygame's window where the drawing will occur.
        :type window: pygame's window.
        :param pose: current pose of the robot.
        :type pose: Pose.
        """
        DrawingUtils.draw_polygon_on_screen(window, RobotSprite.transform(pose, self.left_wheel), self.wheel_color, 0)
        DrawingUtils.draw_polygon_on_screen(window, RobotSprite.transform(pose, self.right_wheel), self.wheel_color, 0)
        DrawingUtils.draw_polygon_on_screen(window, RobotSprite.transform(pose, self.body), self.robot_color, 0)
        DrawingUtils.draw_polygon_on_screen(window, RobotSprite.transform(pose, self.line_sensor), self.robot_color, 0)


class Simulation:
    """
    Represents the simulation.
    """
    def __init__(self, line_follower, track):
        """
        Creates the simulation.

        :param line_follower: the line follower robot.
        :type line_follower: LineFollower.
        :param track: the line track.
        :type track: Track.
        """
        self.line_follower = line_follower
        self.track = track
        start = self.track.get_initial_point()
        self.line_follower.reset(Pose(start.x, start.y, 0.0))
        self.point_list = []  # To draw the robot's path
        # Defining the sprite parameters
        sprite_params = Params()
        sprite_params.wheel_thickness = 0.01
        sprite_params.sensor_thickness = 0.02
        sprite_params.wheel_radius = line_follower.wheel_radius
        sprite_params.wheels_distance = line_follower.wheels_distance
        sprite_params.sensor_offset = line_follower.sensor_offset
        sprite_params.array_width = line_follower.line_sensor.array_width
        # Creating the robot sprite
        self.sprite = RobotSprite(sprite_params)

    def reset(self, controller_params=None):
        """
        Resets the simulation.
        Changing controller parameters is optional. If no controller parameters is passed when calling this
        method, the previous controller parameters will be maintained.

        :param controller_params: new controller parameters.
        :type controller_params: Params.
        """
        start = self.track.get_initial_point()
        self.line_follower.reset(Pose(start.x, start.y, 0.0), controller_params)
        self.point_list = []

    def update_line_sensor_intensity(self):
        """
        Updates the line sensor intensity.
        """
        sensor_range = self.line_follower.line_sensor.sensor_range
        sensor_positions = self.line_follower.get_sensors_global_positions()
        intensity = [0.0] * len(sensor_positions)
        for i in range(len(sensor_positions)):
            for piece in self.track.pieces:
                closest = piece.get_closest_to_point(sensor_positions[i])
                distance = sensor_positions[i].distance(closest)
                intensity[i] = max(intensity[i],
                                   max(0.0, (sensor_range - distance) / sensor_range))
        self.line_follower.set_line_sensor_intensity(intensity)

    def update(self):
        """
        Updates the simulation.
        """
        self.update_line_sensor_intensity()
        self.line_follower.update()
        self.point_list.append((m2pix(self.line_follower.pose.position.x), m2pix(self.line_follower.pose.position.y)))

    def evaluate(self):
        """
        Evaluates the current robot situation.

        :return: reward for the current robot situation.
        :rtype: float.
        """
        linear, angular = self.line_follower.get_velocity()  # v_k, omega_k
        error, detection = self.line_follower.line_sensor.get_error()  # e_k
        track_tangent = self.track.get_tangent(self.line_follower.pose.position)  # t_k
        robot_direction = Vector2(cos(self.line_follower.pose.rotation), sin(self.line_follower.pose.rotation))  # r_k
        dot_product = track_tangent.dot(robot_direction)  # dot(r_k, t_k)
        # rewardk=vk*dot(rk,tk)-w*|ek|

        return linear * dot_product - abs(error) * .5

    def draw(self, window):
        """
        Draws the simulation (line follower robot and track).

        :param window: pygame's window where the drawing will occur.
        :type window: pygame's window.
        """
        self.track.draw(window)
        if len(self.point_list) >= 2:
            pygame.draw.lines(window, (255, 0, 0), False, self.point_list, 4)
        self.sprite.draw(window, self.line_follower.pose)
        sensor_positions = self.line_follower.get_sensors_global_positions()
        for sensor_position in sensor_positions:
            DrawingUtils.draw_circle_on_screen(window, (sensor_position.x, sensor_position.y), 0.01, (255, 0, 0), 0)
