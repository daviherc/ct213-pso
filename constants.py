# Simulation Parameters
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PIX2M = 3.0 / 800.0  # factor to convert from pixels to meters
M2PIX = 1.0 / PIX2M  # factor to convert from meters to pixels
ROBOT_SAMPLE_TIME = 1.0 / 30.0  # Sample time of the robot controller
SIMULATION_FREQUENCY = 60.0  # Frequency of simulation
SIMULATION_SAMPLE_TIME = 1.0 / SIMULATION_FREQUENCY
DRAW_FREQUENCY = 60.0  # Screen update frequency
MAX_ACCELERATED_FACTOR = 200  # How much faster than realtime the simulation is executed in accelerated mode
DEFAULT_ACCELERATED_FACTOR = MAX_ACCELERATED_FACTOR
MAX_EPISODE_TIME = 15.0  # Time limit of a training episode
DETECTION_THRESHOLD = 1.0e-3  # Intensity threshold to consider if a line was detected
ROBOT_CONTROL_DELAY = 2  # Delay between the controller issuing a command and the wheels' references being changed
