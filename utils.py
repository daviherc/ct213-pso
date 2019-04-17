import pygame
from math import sqrt, atan2, cos, sin, pi, fabs
from constants import M2PIX


class Params:
    """
    Represents an auxiliary class for storing parameters.
    I know this is bad hack, but we are using Python anyway.
    """
    pass


def normalize_angle(angle):
    """
    Normalizes an angle to make it be within the range (-pi, pi].

    :param angle: angle to be normalized.
    :type angle: float.
    :return: normalized angle.
    :rtype: float.
    """
    while angle >= pi:
        angle -= 2.0 * pi
    while angle < -pi:
        angle += 2.0 * pi
    return angle


def m2pix(value):
    """
    Converts from meters to pixels.

    :param value: value in meters.
    :type value: float.
    :return: value in pixels.
    :type value: float.
    """
    return round(M2PIX * value)


def clamp(value, minimum, maximum):
    """
    Clamps a value to keep it within the interval [minimum, maximum].

    :param value: value to be clamped.
    :type value: float.
    :param minimum: minimum value.
    :type minimum: float.
    :param maximum: maximum value.
    :type maximum: float.
    :return: clamped value.
    :rtype: float.
    """
    if value > maximum:
        return maximum
    elif value < minimum:
        return minimum
    return value


class DrawingUtils:
    """
    Represents an auxiliary class for drawing.
    """
    @staticmethod
    def rectangle_to_polygon(rectangle):
        """
        Converts a rectangle to a polygon.

        :param rectangle: rectangle as the following tuple: (left, top, width, height).
        :type rectangle: four-dimensional tuple of floats.
        :return: polygon as a list containing the rectangle's points: [top_left, top_right, bottom_right, bottom_left].
        :rtype: list of two-dimensional tuples.
        """
        top_left = (rectangle[0], rectangle[1])
        top_right = (rectangle[0] + rectangle[2], rectangle[1])
        bottom_right = (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3])
        bottom_left = (rectangle[0], rectangle[1] + rectangle[3])
        return [top_left, top_right, bottom_right, bottom_left]

    @staticmethod
    def draw_polygon_on_screen(window, points, color, width):
        """
        Draws a polygon on screen. The measurement unit is meters.

        :param window: pygame's window where the drawing will occur.
        :type window: pygame's window.
        :param points: points of the polygon.
        :type points: list of two-dimensional tuples.
        :param color: polygon's color in RGB format.
        :type color: three-dimensional tuple of ints.
        :param width: thickness of the polygon line.
        :type width: int.
        """
        points_on_screen = []
        for point in points:
            points_on_screen.append((m2pix(point[0]), m2pix(point[1])))
        pygame.draw.polygon(window, color, points_on_screen, width)

    @staticmethod
    def draw_rectangle_on_screen(window, origin, dimensions, color, width):
        """
        Draws a rectangle on screen. The measurement unit is meters.

        :param window: pygame's window where the drawing will occur.
        :type window: pygame's window.
        :param origin: the rectangle origin.
        :type origin: two-dimensional tuple of floats.
        :param dimensions: the rectangle dimensions (width and height).
        :type dimensions: two-dimensional tuple of floats.
        :param color: circle's color in RGB format.
        :type color: three-dimensional tuple of ints.
        :param width: thickness of the rectangle line.
        :type width: int.
        """
        origin_on_screen = (round(M2PIX * origin[0]), round(M2PIX * origin[1]))
        dimensions_on_screen = (round(M2PIX * dimensions[0]), round(M2PIX * dimensions[1]))
        pygame.draw.rect(window, color, pygame.Rect(origin_on_screen, dimensions_on_screen), width)

    @staticmethod
    def draw_circle_on_screen(window, center, radius, color, width):
        """
        Draws a circle on screen. The measurement unit is meters.

        :param window: pygame's window where the drawing will occur.
        :type window: pygame's window.
        :param center: the circle center.
        :type center: two-dimensional tuple of floats.
        :param radius: the circle radius.
        :type radius: float.
        :param color: circle's color in RGB format.
        :type color: three-dimensional tuple of ints.
        :param width: thickness of the circle line.
        :type width: int.
        """
        center_on_screen = (int(round(M2PIX * center[0])), int(round(M2PIX * center[1])))
        radius_on_screen = round(M2PIX * radius)
        pygame.draw.circle(window, color, center_on_screen, radius_on_screen, width)

    @staticmethod
    def draw_line_on_screen(window, start, end, color, width):
        """
        Draws a line on screen. The measurement unit is meters.

        :param window: pygame's window where the drawing will occur.
        :type window: pygame's window.
        :param start: the line's start.
        :type start: two-dimensional tuple of floats.
        :param end: the line's end.
        :type end: two-dimensional tuple of floats.
        :param color: line's color in RGB format.
        :type color: three-dimensional tuple of ints.
        :param width: thickness of the line.
        :type width: int.
        """
        start_on_screen = (round(M2PIX * start[0]), round(M2PIX * start[1]))
        end_on_screen = (round(M2PIX * end[0]), round(M2PIX * end[1]))
        pygame.draw.line(window, color, start_on_screen, end_on_screen, width)

    @staticmethod
    def draw_arc_on_screen(window, center, radius, start_angle, stop_angle, color, width):
        """
        Draws a arc on screen. The measurement unit is meters.

        :param window: pygame's window where the drawing will occur.
        :type window: pygame's window.
        :param center: the arc center.
        :type center: two-dimensional tuple of floats.
        :param radius: the arc radius.
        :type radius: float.
        :param radius: the arc start angle.
        :type radius: float.
        :param radius: the arc stop angle.
        :type radius: float.
        :param color: arc's color in RGB format.
        :type color: three-dimensional tuple of ints.
        :param width: thickness of the arc line.
        :type width: int.
        """
        top_left_on_screen = (m2pix(center[0] - radius), m2pix(center[1] - radius))
        diameter_on_screen = m2pix(2.0 * radius)
        rectangle_on_screen = (top_left_on_screen[0], top_left_on_screen[1],
                               diameter_on_screen, diameter_on_screen)
        pygame.draw.arc(window, color, rectangle_on_screen, start_angle, stop_angle + 0.01, width)


class Vector2:
    """
    Represents a bidimensional geometric vector.
    """
    def __init__(self, x, y):
        """
        Creates a bidimensional geometric vector.

        :param x: x coordinate.
        :type x: float.
        :param y: y coordinate.
        :type y: float.
        """
        self.x = x
        self.y = y

    def __add__(self, other):
        """
        Sums two vectors.

        :param other: the other vector this vector will be added to.
        :type other: Vector2.
        :return: the result of the operation.
        :rtype: Vector2.
        """
        result = Vector2(self.x, self.y)
        result.x += other.x
        result.y += other.y
        return result

    def __sub__(self, other):
        """
        Subtracts two vectors.

        :param other: the other vector this vector will be subtracted to.
        :type other: Vector2.
        :return: the result of the operation.
        :rtype: Vector2.
        """
        result = Vector2(self.x, self.y)
        result.x -= other.x
        result.y -= other.y
        return result

    def __mul__(self, scalar):
        """
        Multiplies this vector by a scalar.

        :param scalar: the scalar used in the operation.
        :type scalar: float.
        :return: the result of the operation.
        :rtype: Vector2.
        """
        result = Vector2(self.x, self.y)
        result.x *= scalar
        result.y *= scalar
        return result

    def normalize(self):
        """
        Normalizes this vector, i.e. make it have unit norm.
        """
        norm = self.length()
        self.x /= norm
        self.y /= norm

    def length(self):
        """
        Computes the length of this vector.

        :return: the length of this vector.
        :rtype: float.
        """
        return sqrt(self.x * self.x + self.y * self.y)

    def distance(self, other):
        """
        Computes the distance from this vector to another vector.

        :param other: the other vector whose distance to this vector will be computed.
        :type other: Vector2.
        :return: the distance between the two vectors.
        :rtype: float.
        """
        diff = self - other
        return diff.length()

    def dot(self, other):
        """
        Computes the dot product of two vectors.

        :param other: the other vector used in the operation.
        :rtype other: Vector2.
        :return: the dot product of the two vectors.
        :rtype: float.
        """
        return self.x * other.x + self.y * other.y

    def to_tuple(self):
        """
        Transforms this vector into a tuple.

        :return: vector as tuple (x, y).
        :rtype: two-dimensional tuple of floats.
        """
        ret = (self.x, self.y)
        return ret


class Pose:
    """
    Represents a pose on the plane, i.e. a (x, y) position plus a rotation.
    """
    def __init__(self, x, y, rotation):
        """
        Creates a pose on the plane.

        :param x: x coordinate.
        :type x: float.
        :param y: y coordinate.
        :type y: float.
        :param rotation: rotation around z axis.
        :type rotation: float.
        """
        self.position = Vector2(x, y)
        self.rotation = rotation


class LineSegment:
    """
    Represents a line segment.
    """
    def __init__(self, start, end):
        """
        Creates a line segment.

        :param start: start point of the line segment.
        :type start: Vector2.
        :param end: end point of the line segment.
        :type end: Vector2.
        """
        self.start = start
        self.end = end
        self.length = start.distance(end)

    def get_length(self):
        """
        Obtains the length of the line segment.

        :return: the length of the line segment.
        :rtype: float.
        """
        return self.length

    def interpolate(self, t):
        """
        Interpolates the start and end points of the line segment to obtain an intermediary point.

        :param t: interpolation parameter (goes from 0 to 1).
        :rtype t: float.
        :return: interpolated point.
        :rtype: Vector2
        """
        return self.start + (self.end - self.start) * t

    def get_tangent(self, reference_point):
        """
        Obtains the tangent of this line segment at a reference point.

        :param reference_point: reference point used in the computation.
        :type reference_point: Vector2.
        :return: the tangent of the line segment given the reference point.
        :rtype: Vector2.
        """
        # The tangent of a line segment does not depend on the point we are computing it
        diff = self.end - self.start
        diff.normalize()
        return diff

    def get_closest_to_point(self, point):
        """
        Obtains the closest point in the line segment to a given point.

        :param point: point used as reference.
        :type point: Vector2.
        :return: closest point in the line segment.
        :rtype: Vector2.
        """
        sp = point - self.start
        se = self.end - self.start
        t = se.dot(sp) / se.dot(se)
        # If the closest in the line is outside the segment, then the closest is the start or end points.
        if t < 0.0:
            return self.start
        elif t > 1.0:
            return self.end
        else:
            closest = self.start + se * t
            return closest


class Arc:
    """
    Represents an arc.
    """
    def __init__(self, center, radius, start_angle, stop_angle):
        """
        Creates an arc.

        :param center: arc center.
        :type center: float.
        :param radius: arc radius.
        :type radius: float.
        :param start_angle: arc start angle.
        :type start_angle: float.
        :param stop_angle: arc stop angle.
        :type stop_angle: float.
        """
        self.center = center
        self.radius = radius
        self.start_angle = start_angle
        self.stop_angle = stop_angle
        self.length = fabs(normalize_angle(stop_angle - start_angle)) * self.radius

    def get_length(self):
        """
        Obtains the length of the arc.

        :return: the length of the arc.
        :rtype: float.
        """
        return self.length

    def interpolate(self, t):
        """
        Interpolates the start and stop angles of the line segment to obtain an intermediary point.

        :param t: interpolation parameter (goes from 0 to 1).
        :rtype t: float.
        :return: interpolated point.
        :rtype: Vector2
        """
        angle = normalize_angle(self.start_angle + t * normalize_angle(self.stop_angle - self.start_angle))
        x = self.center.x + self.radius * cos(angle)
        y = self.center.y + self.radius * sin(angle)
        return Vector2(x, y)

    def get_tangent(self, reference_point):
        """
        Obtains the tangent of this arc at a reference point.

        :param reference_point: reference point used in the computation.
        :type reference_point: Vector2.
        :return: the tangent of arc given the reference point.
        :rtype: Vector2.
        """
        direction = reference_point - self.center
        angle = atan2(direction.y, direction.x)
        angle_diff = self.stop_angle - self.start_angle
        if angle_diff >= 0.0:
            return Vector2(-sin(angle), cos(angle))
        return Vector2(sin(angle), -cos(angle))

    def get_closest_to_point(self, point):
        """
        Obtains the closest point in the arc to a given point.

        :param point: point used as reference.
        :type point: Vector2.
        :return: closest point in the arc.
        :rtype: Vector2.
        """
        direction = point - self.center
        angle = atan2(direction.y, direction.x)
        if normalize_angle(angle - self.start_angle) < 0.0:
            angle = self.start_angle
        elif normalize_angle(angle - self.stop_angle) > 0.0:
            angle = self.stop_angle
        x = self.center.x + self.radius * cos(angle)
        y = self.center.y + self.radius * sin(angle)
        return Vector2(x, y)
