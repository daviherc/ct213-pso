from utils import LineSegment, Arc
from math import inf
from utils import DrawingUtils


class Track:
    """
    Represents a line track for a line follower robot.
    """
    def __init__(self):
        """
        Creates the line track.
        """
        self.pieces = []
        self.total_length = 0.0

    def get_initial_point(self):
        """
        Obtains the track's initial position.

        :return: track's initial position.
        :rtype: Vector2.
        """
        if isinstance(self.pieces[0], LineSegment):
            return self.pieces[0].start

    def add_line_piece(self, start, end):
        """
        Adds a line segment piece to the track. Notice that start and end define the expected
        "direction" of transversing.

        :param start: line begin.
        :type start: Vector2.
        :param end: line end.
        :type end: Vector2.
        """
        self.pieces.append(LineSegment(start, end))
        self.total_length += self.pieces[-1].get_length()

    def add_arc_piece(self, center, radius, start_angle, stop_angle):
        """
        Adds a arc piece to the track. Notice that start_angle and stop_angle define the expected
        "direction" of transversing.

        :param center: arc center.
        :type center: float.
        :param radius: arc radius.
        :type radius: float.
        :param start_angle: arc start angle.
        :type start_angle: float.
        :param stop_angle: arc stop angle.
        :type stop_angle: float.
        """
        self.pieces.append(Arc(center, radius, start_angle, stop_angle))
        self.total_length += self.pieces[-1].get_length()

    def get_tangent(self, reference_point):
        """
        Obtains the tangent of the closest point in the track to a given point.

        :param reference_point: point used as reference to compute the tangent in the track.
        :type reference_point: Vector2.
        :return: tangent of the closest point in the track.
        :rtype: Vector2.
        """
        closest_distance = inf
        current_piece = None
        # Iterate over all track's pieces to find the piece which is closest to the reference point.
        for piece in self.pieces:
            closest = piece.get_closest_to_point(reference_point)
            distance = closest.distance(reference_point)
            if distance < closest_distance:
                closest_distance = distance
                current_piece = piece
        # Returns the tangent of the closest point to the reference point in this piece
        return current_piece.get_tangent(reference_point)

    def draw(self, window):
        """
        Draws the track.

        :param window: pygame's window where the drawing will occur.
        :type window: pygame's window.
        """
        for piece in self.pieces:
            if isinstance(piece, LineSegment):
                DrawingUtils.draw_line_on_screen(window, piece.start.to_tuple(), piece.end.to_tuple(), (0, 0, 0), 2)
            elif isinstance(piece, Arc):
                DrawingUtils.draw_arc_on_screen(window, piece.center.to_tuple(), piece.radius, piece.start_angle,
                                                piece.stop_angle, (0, 0, 0), 2)
