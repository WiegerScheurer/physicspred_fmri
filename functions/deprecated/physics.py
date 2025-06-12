import numpy as np
import math
import matplotlib.pyplot as plt

# Probably deprecated
def check_collision(ball_pos, line_angle, ball):
    """Returns True if the ball's edge intersects the diagonal line."""
    x, y = ball_pos
    if line_angle[:2] == "45":  
        return abs(y - x) <= 0 # ball.radius  # Ball touches y = x
    elif line_angle[:3] == "135":
        return abs(y + x) <= 0 # ball.radius  # Ball touches y = -x
    return False  # No line


def collide(start_direction: str, line_angle: int, ball_speed: float):
    """Returns the new direction vector of the ball after a collision.
    
    N.B.: Start direction is the direction the ball is moving towards, 
    so not the start location of the ball.
    """
    direction_angles = {
        "up": 270,
        "down": 90,
        "left": 0,
        "right": 180
    }
    
    # Get the initial angle of the ball's movement
    initial_angle = direction_angles[start_direction]

    # Calculate the angle of incidence
    if line_angle in [45, 135]:
        normal_angle = line_angle
    else:
        return start_direction
    
    # Calculate the angle of reflection
    angle_of_reflection = (2 * normal_angle - initial_angle) % 360
    
    # Convert the angle of reflection back to a direction vector
    new_direction = (
        math.cos(math.radians(angle_of_reflection)) * ball_speed,
        math.sin(math.radians(angle_of_reflection)) * ball_speed
    )
    
    return new_direction

# Function to calculate the decay factor
def calculate_decay_factor(start_speed, elapsed_time, total_time, constant = (0.01/6)):
    # constant = 0.01/6 # Based on what my eyes see as a realistic decay
    # constant = 1
    decay_rate = start_speed * constant  # Adjust this value to control the rate of decay
    return np.exp(-decay_rate * (elapsed_time / total_time))

def will_cross_fixation(ball_pos, velocity, skip_factor):
    # Calculate the next position
    next_pos = ball_pos + np.array([velocity[0] * skip_factor, velocity[1] * skip_factor])    
    
    # Check if the line segment between ball_pos and next_pos crosses the (0, 0) point
    # This can be done by checking if the signs of the coordinates change
    if (np.sign(ball_pos[0]) != np.sign(next_pos[0]) or np.sign(ball_pos[1]) != np.sign(next_pos[1])):
        return True
    return False

def compute_speed(direction_vector):
    """Computes the speed from the direction vector."""
    dx, dy = direction_vector
    speed = math.sqrt(dx**2 + dy**2)
    return speed

def change_speed(ball_speed: float, change_factor: float, direction: str):
    """Returns the new direction vector of the ball after a speed change."""
    dy, dx = _dir_to_vec(direction)
    new_speed = ball_speed * change_factor
    new_direction = (dx * new_speed, dy * new_speed)
    return new_direction

def velocity_to_direction(velocity):
    """Converts a velocity vector to a direction string."""
    x, y = velocity
    if abs(x) > abs(y):
        return "left" if x < 0 else "right"
    else:
        return "down" if y < 0 else "up"  # Vertical axis is inverted
    
    
def _dir_to_vec(direction:str) -> tuple:
    """Turn direction string into vector representation
    Args:
        direction (str): The direction string
    Returns:
        tuple: The vector representation in row,column (y,x)
    """    
    directions = {"up": (1, 0),
              "right": (0, 1), 
              "down": (-1, 0),
              "left": (0, -1)}
    
    return directions[direction]

def _vec_to_dir(vector:tuple) -> str:
    """Turn vector representation into direction string
    Args:
        vector (tuple): The vector representation in row,column (y,x)
    Returns:
        str: The direction string
    """    
    vectors = {(1, 0): "up",
              (0, 1): "right", 
              (-1, 0): "down",
              (0, -1): "left"}
    
    return vectors[vector]

## Not really needed though, locations are on a 3x3 grid
def _dir_to_loc(direction:str) -> tuple:
    """Turn direction string into location representation
    Args:
        direction (str): The direction string
    Returns:
        tuple: The location representation in row,column (y,x)
    """    
    locations = {"up": (0, 1),
              "right": (1, 2), 
              "down": (2, 1),
              "left": (1, 0)}
    
    return locations[direction]

def _dir_to_velocity(direction: str | tuple, speed:float) -> tuple:
    """Turn direction string into velocity representation
    Args:
        direction (str): The direction string
        speed (float): The speed of the object
    Returns:
        tuple: The velocity representation in row,column (y,x)
    """    
    direction_vector = _dir_to_vec(direction) if isinstance(direction, str) else direction
    velocity_vector = tuple(speed * dir_axis for dir_axis in direction_vector)
    
    return np.array((velocity_vector[1], velocity_vector[0]))

def _rotate_90(start_direction, left_or_right):
    """
    Rotate a 2D vector by 90 degrees in the specified direction.
    
    Args:
        start_direction (tuple): The initial direction vector as a tuple of two elements.
        left_or_right (str): The direction to rotate, either "left" or "right".
    
    Returns:
        tuple: The rotated direction vector as a tuple of two elements (y, x) or (row, column).
    """
    if type(start_direction) != tuple:
        start_direction = _dir_to_vec(start_direction)
    
    # Define the rotation matrix for 90 degrees
    rotation_matrix_90 = np.array([[0, -1], [1, 0]])
    rotation_matrix_270 = np.array([[0, 1], [-1, 0]])

    towards = {"left": rotation_matrix_270,
               "right": rotation_matrix_90}

    # Convert the direction tuple to a numpy array
    direction_vector = np.array(start_direction)
    
    # Perform the matrix multiplication to rotate the vector
    rotated_vector = np.dot(towards[left_or_right], direction_vector)
    # Convert the result back to a tuple and return
    return tuple(rotated_vector)

def _flip_dir(direction: str | tuple) -> str | tuple:
    """Where does the ball end up, given a direction? Assuming a continuous path, so no collision anymore.
        This basically just flips the direction value to the opposite on the relevant axis.

    Args:
        direction (str | tuple): Where does the ball go?

    Returns:
        str | tuple: The opposite point of the field.
    """    
    if isinstance(direction, str):
        flipped_dir = _vec_to_dir(tuple(dir_axis * -1 for dir_axis in _dir_to_vec(direction)))
    elif isinstance(direction, tuple):
        flipped_dir = tuple(dir_axis * -1 for dir_axis in direction)
    else:
        raise ValueError("Direction must be either a string or a tuple")
    
    return flipped_dir

# def _bounce_ball(start_direction: str, interactor: str):
#     """
#     Bounces a ball based on the start direction and the type of interactor.

#     Parameters:
#     start_direction (str): The initial direction of the ball.
#     interactor (str): The type of interactor.

#     Returns:
#     str: The new direction of the ball after bouncing.

#     """

#     if interactor == "45":
#         relative_direction = "left" if start_direction in ["right", "left"] else "right"
#         end_loc = _rotate_90(start_direction=start_direction, left_or_right=relative_direction)
#     elif interactor == "135":
#         relative_direction = "left" if start_direction in ["up", "down"] else "right"
#         end_loc = _rotate_90(start_direction=start_direction, left_or_right=relative_direction)
#     else:
#         end_loc = _dir_to_vec(start_direction) # When no interactor, ball ends up in the same direction
        
#     end_direction = _flip_dir(end_loc)
    
#     return _vec_to_dir(end_direction)

# def _bounce_ball(start_direction: str, interactor: str, str_or_tuple_out:str = "str"):
#     """
#     Bounces a ball based on the start direction and the type of interactor.

#     Parameters:
#     start_direction (str): The initial direction of the ball.
#     interactor (str): The type of interactor.

#     Returns:
#     str: The new direction of the ball after bouncing.

#     """

#     if interactor[:2] == "45":
#         relative_direction = "left" if start_direction in ["right", "left"] else "right"
#         end_loc = _rotate_90(start_direction=start_direction, left_or_right=relative_direction)
#     elif interactor[:3] == "135":
#         relative_direction = "left" if start_direction in ["up", "down"] else "right"
#         end_loc = _rotate_90(start_direction=start_direction, left_or_right=relative_direction)
#     else:
#         end_loc = _dir_to_vec(start_direction) # When no interactor, ball ends up in the same direction
        
#     end_direction = _flip_dir(end_loc)
    
#     return end_direction if str_or_tuple_out == "tuple" else _vec_to_dir(end_direction)

def _bounce_ball(start_direction: str, interactor: str, str_or_tuple_out:str = "str"):
    """
    Bounces a ball based on the start direction and the type of interactor.

    Parameters:
    start_direction (str): The initial direction of the ball.
    interactor (str): The type of interactor.

    Returns:
    str: The new direction of the ball after bouncing.

    """

    if interactor[:3] == "135":
        relative_direction = "left" if start_direction in ["right", "left"] else "right"
        end_loc = _rotate_90(start_direction=start_direction, left_or_right=relative_direction)
    elif interactor[:2] == "45":
        relative_direction = "left" if start_direction in ["up", "down"] else "right"
        end_loc = _rotate_90(start_direction=start_direction, left_or_right=relative_direction)
    else:
        end_loc = _dir_to_vec(start_direction) # When no interactor, ball ends up in the same direction
        
    end_direction = _flip_dir(end_loc)
    
    return end_direction if str_or_tuple_out == "tuple" else _vec_to_dir(end_direction)


def plot_positions(start_pos, end_pos, pred_to_input, interactor):
    positions = {"up": (0, 1), "right": (1, 0), "down": (0, -1), "left": (-1, 0)}
    
    fig, ax = plt.subplots()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    
    # Plot start position
    start_coords = positions[start_pos]
    ax.plot(start_coords[0], start_coords[1], 'go', markersize=10, label='Start Position', alpha=.5)
    
    # Plot end position
    end_coords = positions[end_pos]
    ax.plot(end_coords[0], end_coords[1], 'ro', markersize=10, label='End Position', alpha=.5)
    
    # Plot predicted positions
    for pos, value in pred_to_input.items():
        if value[0] == 1:
            pred_coords = positions[pos]
            ax.plot(pred_coords[0], pred_coords[1], 'bo', markersize=10, label='Predicted Position', alpha=.5)
    
    # Add diagonal stripe based on interactor value
    if interactor[:2] == "45":
        ax.plot([-.2, .2], [-.2, .2], 'k-', label='45° interactor')
    elif interactor[:3] == "135":
        ax.plot([-.2, .2], [.2, -.2], 'k-', label='135° interactor')
    
    ax.legend()
    ax.axis("off")
    plt.show()

def predict_ball_path(hypothesis: str, interactor: str, start_pos: str, end_pos: str, plot: bool = False):
    """
    Predict the path of a ball based on the given parameters.
    
    Args:
        hypothesis (str): The predictor hypothesis, either "abs" or "rel".
        interactor (str): The interactor hypothesis, either "none" or "abs".
        start_pos (str): The starting position of the ball.
        end_pos (str): The ending position of the ball.
        plot (bool): Whether to plot the positions or not.
    
    Returns:
        dict: A dictionary representing the path of the ball.
    """
    pred_to_input = {"up": [0],
                     "right": [0],
                     "down": [0],
                     "left": [0]}
    
    # NOTE: Predictions are about the ball direction AFTER collision, so it's 0 for start positions
    if hypothesis == "abs":
        pred_to_input[_flip_dir(start_pos)] = [1] # Opposite of start position
        
    elif hypothesis == "sim":
        # NOTE: flip_dir is used to get the ball direction based on the start location
        predicted_dir = _bounce_ball(start_direction=_flip_dir(start_pos), interactor=interactor)
        predicted_endloc = _flip_dir(predicted_dir) # Flip direction to get endpoint
        pred_to_input[predicted_endloc] = [1]
        
    for receptive_field in pred_to_input.keys():
        pred_to_input[receptive_field].append(0) # Add column for sensory input
    pred_to_input[end_pos][1] = 1 # Change to 1 for end position

    # Turn the dictionary list values into tuples    
    pred_to_input_tuples = {key: tuple(value) for key, value in pred_to_input.items()}

    if plot:
        plot_positions(start_pos, end_pos, pred_to_input, interactor)
    
    return pred_to_input_tuples

def get_dist_dif(ball_radius):
    """
    Calculate the difference in distance between a bouncing ball
    and a continuous ball, based on the horizontal side of the
    triangle formed by the ball when hitting a 45-degree angle.
    
    Parameters:
        ball_radius (float): The radius of the ball.
    
    Returns:
        float: The the less distance traveled by the bouncing ball.
    """
    return np.cos(np.deg2rad(45)) * ball_radius
##### DOES THE EXACT SAME AS BELOW, BUT WITH TRIGONOMETRIC FUNCTIONS#####
def get_bounce_dist(ball_radius):
    """Compute the horizontal/vertical distance from ball center to 
    contactpoint of the interactor. These are the two right sides of 
    the triangle, where the diagonal is the ball radius.
    N.B.: As the triangle is an isosceles (gelijkbenig) triangle, the two 
    sides are equal.
    N.B.: As the angle is 45 degrees, we can just use the eenheidscirkel
    coordinates for 45 degrees = (sqrt(2) / 2)
    

    Args:
        ball_radius (float): radius of the ball

    Returns:
        float: the contactpoint coordinates between ball and interactor.
    """        
    return ball_radius * (np.sqrt(2) / 2)