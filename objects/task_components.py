import os
import sys
import yaml
import random
import colour
import numpy as np
from psychopy import visual, gui, core, data, filters
from omegaconf import OmegaConf



sys.path.append(
    "/Users/wieger.scheurer/repoclones/physicspred"
)  # To enable importing from repository folders
# sys.path.append(
#     "/Users/wiegerscheurer/repos/physicspred"
# )  # To enable importing from repository folders

from functions.physics import get_bounce_dist
from functions.utilities import oklab_to_rgb

# Whether running on Mac or lab computer
mac_or_lab = "lab" # mac or lab

# Load configuration from YAML file
# config_path = os.path.join(os.path.dirname(__file__), os.pardir, "config_lumin.yaml")
config_path = os.path.join(os.path.dirname(__file__), os.pardir, "fmri_settings.yml")
# config_path = os.path.join(os.path.dirname(__file__), os.pardir, "behav_settings.yml")
config = OmegaConf.load(config_path)

# occluder_type = config.occluder.type
win_dims = config.display.win_dims
ball_radius = config.ball.radius
# ball_speed = config.ball.avg_speed
ball_start_color_mean = config.ball.start_color_mean
interactor_height = config.interactor.height
interactor_width = config.interactor.width
# int_45_path = config.paths.int_45 # if mac_or_lab == "mac" else config.paths.lab_int_45 # TODO: ADAPT
# int_135_path = config.paths.int_135 # if mac_or_lab == "mac" else config.paths.lab_int_135 # TODO: ADAPT
int_45_path = config.interactor.path_45
int_135_path = config.interactor.path_135


occluder_radius = config.occluder.radius
occluder_opacity = config.occluder.opacity
background_luminance = config.display.background_luminance
verbose = config.experiment.verbose
square_size = config.display.square_size
exp_parameters = config.experiment.exp_parameters
full_screen = config.display.full_screen
experiment_screen = config.display.experiment_screen
draw_grid = config.display.draw_grid
fixation_color = config.fixation.color
fixation_length = config.fixation.length
fixation_thickness = config.fixation.thickness

exp_data = {par: [] for par in exp_parameters}

if __name__ == "__main__":

    win = visual.Window(
        size=win_dims,        # The size of the window in pixels (width, height).
        fullscr=full_screen,  # Whether to run in full-screen mode. Overrides size arg
        screen=experiment_screen,  # The screen number to display the window on (0 is usually the primary screen).
        winType="pyglet",  # The backend to use for the window (e.g., 'pyglet', 'pygame').
        allowStencil=False,  # Whether to allow stencil buffer (used for advanced graphics).
        # monitor='testMonitor',    # The name of the monitor configuration to use (defined in the Monitor Center).
        # color= [background_luminance] * 3,  # [0, 0, 0],          # The background color of the window (in RGB space).
        # color=[0, 0, 0],          # The background color of the window (in RGB space). # doesn't matter anymore
        color=[-.75, -.75, -.75],          # The background color of the window (in RGB space). # doesn't matter anymore
        colorSpace="rgb",  # The color space for the background color (e.g., 'rgb', 'dkl', 'lms').
        backgroundImage="",  # Path to an image file to use as the background.
        backgroundFit="none",  # How to fit the background image ('none', 'fit', 'stretch').
        blendMode="avg",  # The blend mode for drawing (e.g., 'avg', 'add').
        useFBO=True,  # Whether to use Frame Buffer Objects (for advanced graphics).
        units="pix",  # The default units for window operations (e.g., 'pix', 'norm', 'cm', 'deg', 'height').
        multiSample=False,  # Whether to use multi-sample anti-aliasing. # QUADRUPLE CHECK IF THIS DOESN'T MESS UP MOVEMENT OF BALL, APPEARED TO BE SLUGGISH
    )

    win_dims = win.size

    fixation = visual.TextStim(win, text="+", color=fixation_color, pos=(0, 0), height=50)

# Create checkerboard pattern
def create_checkerboard(size, check_size, light_color, dark_color):
    pattern = np.zeros((size, size, 3))
    num_checks = size // check_size
    for i in range(num_checks):
        for j in range(num_checks):
            color = light_color if (i + j) % 2 == 0 else dark_color
            pattern[i*check_size:(i+1)*check_size, j*check_size:(j+1)*check_size] = color
    return pattern

# Set luminance values in Oklab space
mean_ball_color = ball_start_color_mean
checker_light = oklab_to_rgb([mean_ball_color + 0.05, 0, 0])
checker_dark = oklab_to_rgb([mean_ball_color - 0.05, 0, 0])

# Make the lab color beige

# Create checkerboard texture
check_size = 4  # Size of each check in pixels
texture = create_checkerboard(square_size, check_size, checker_light, checker_dark)

# Create grating stimulus using ImageStim
grating = visual.ImageStim(
    win=win,
    image=texture,
    size=[square_size, square_size],
    units='pix'
)

# # Create checkerboard texture TO CONTROL FOR CONTRAST WITH BACKGROUND
# size = square_size  # Size of the texture in pixels
# check_size = 8  # Size of each check in pixels
# texture = np.ones((size, size, 3))
# checks = np.zeros((size, size))
# checks[::check_size, ::check_size] = 1
# checks[check_size::check_size, check_size::check_size] = 1

# # Set luminance values in Oklab space
# mean_ball_color = config["ball_start_color_mean"]
# checker_light = mean_ball_color + 0.2
# checker_dark = mean_ball_color - 0.2

# # Convert Oklab to RGB (simplified, you may need a proper conversion function)
# texture[checks == 1] = list(oklab_to_rgb([checker_light, 0, 0], True))  # Lighter checks
# texture[checks == 0] = list(oklab_to_rgb([checker_dark, 0, 0], True))   # Darker checks

# # Create grating stimulus
# grating = visual.GratingStim(
#     win=win,
#     tex=texture,
#     size=[square_size, square_size],
#     units='pix',
#     sf=1.0/check_size
# )

####################################### MAKING A BETTER BALL #############################
ball = visual.Circle(win, 
                     radius=ball_radius, 
                     edges=64,
                     fillColor="white",#config["ball_fillcolor"], 
                     lineColor="white", #config["ball_linecolor"], 
                     interpolate=True,
                     opacity=1)

# Create a 2D isotropic Gaussian
gaussian = visual.GratingStim(
    win=win,
    tex='sin',
    mask='gauss',
    size=(ball_radius*1.5, ball_radius),  # Size in pixels
    sf=0,  # Spatial frequency (0 for no grating)
    contrast=1,
    color='white',
    opacity=.8
)

# Create a 2D isotropic Gaussian
ball_tone = visual.GratingStim(
    win=win,
    tex='sin',
    mask='gauss',
    size=(ball_radius*2.1, ball_radius*2.1),  # Size in pixels
    # size=(ball_radius, ball_radius),  # Size in pixels
    sf=0,  # Spatial frequency (0 for no grating)
    contrast=1,
    color='white',
    opacity=.9,
)

# Create a 2D isotropic Gaussian
ball_glimmer = visual.GratingStim(
    win=win,
    tex='sin',
    mask='gauss',
    size=(ball_radius/1.5, ball_radius*1.5),  # Size in pixels
    sf=0,  # Spatial frequency (0 for no grating)
    contrast=1,
    color='white',
)


# Calculate the offset
offset_y = interactor_width / 2

####################################### Defining the interactor lining #####################################
############################################### Original ###################################################
# Define line_45
line_45 = visual.Rect(
    win,
    width=interactor_width,
    height=interactor_height,
    fillColor="red",
    lineColor="red",
)
line_45.ori = 45  # Rotate the line by 45 degrees
line_45.pos = (offset_y, 0)  # Adjust position

# Define line_135
line_135 = visual.Rect(
    win,
    width=interactor_width,
    height=interactor_height,
    fillColor="red",
    lineColor="red",
)
line_135.ori = 135  # Rotate the line by 135 degrees
line_135.pos = (0, -offset_y)  # Adjust position

############################################### New one ###################################################

def create_interactor(win, width, height, fill_color, line_color, ori, pos):
    rect = visual.Rect(
        win,
        width=width,
        height=height,
        fillColor=fill_color,
        lineColor=line_color,
    )
    rect.ori = ori
    rect.pos = pos
    return rect

# bounce_dist = get_bounce_dist(ball_radius + (interactor_width / 2))
bounce_dist = get_bounce_dist(ball_radius + (interactor_width / 2 * 1.8)) # 1.8 factor is due to the that now we use an image

# These are the old, plain rectangle interactors 

# Define line_45_top and line_45_bottom
# line_45_bottom = create_interactor(win, interactor_width, interactor_height, "red", "red", 45, ((bounce_dist), -bounce_dist))
# line_45_top = create_interactor(win, interactor_width, interactor_height, "red", "red", 45, (-(bounce_dist), bounce_dist))

# Define line_135_top and line_135_bottom
# line_135_bottom = create_interactor(win, interactor_width, interactor_height, "red", "red", 135, (-bounce_dist, -(bounce_dist)))
# line_135_top = create_interactor(win, interactor_width, interactor_height, "red", "red", 135, (bounce_dist, (bounce_dist)))

line_45_bottom = visual.ImageStim(
    win,
    # image="/Users/wiegerscheurer/Stimulus_material/interactor_45_flat_beige.png", 
    image=int_45_path,
    size=(interactor_height, interactor_height),
    pos=(bounce_dist, -(bounce_dist)),
    opacity=1,
    interpolate=True,
)

line_45_top = visual.ImageStim(
    win,
    # image="/Users/wiegerscheurer/Stimulus_material/interactor_45_flat_white.png", 
    image=int_45_path,
    size=(interactor_height, interactor_height),
    pos= (-(bounce_dist), bounce_dist),
    opacity=1,
    interpolate=True,
)

line_135_bottom = visual.ImageStim(
    win,
    # image="/Users/wiegerscheurer/Stimulus_material/interactor_135_flat_white.png", 
    image=int_135_path,
    size=(interactor_height, interactor_height),
    pos=(-bounce_dist, -(bounce_dist)),
    opacity=1,
    interpolate=True,
)

line_135_top = visual.ImageStim(
    win,
    # image="/Users/wiegerscheurer/Stimulus_material/interactor_135_flat_white.png", 
    image=int_135_path,
    size=(interactor_height, interactor_height),
    pos= ((bounce_dist), bounce_dist),
    opacity=1,
    interpolate=True,
)


################### OCCLUDER ######################

cross_factor = occluder_radius / 65
# Define the vertices for a cross shape
cross_vertices = [
    (-occluder_radius, -occluder_radius / cross_factor), (-occluder_radius, occluder_radius / cross_factor),
    (-occluder_radius / cross_factor, occluder_radius / cross_factor), (-occluder_radius / cross_factor, occluder_radius),
    (occluder_radius / cross_factor, occluder_radius), (occluder_radius / cross_factor, occluder_radius / cross_factor),
    (occluder_radius, occluder_radius / cross_factor), (occluder_radius, -occluder_radius / cross_factor),
    (occluder_radius / cross_factor, -occluder_radius / cross_factor), (occluder_radius / cross_factor, -occluder_radius),
    (-occluder_radius / cross_factor, -occluder_radius), (-occluder_radius / cross_factor, -occluder_radius / cross_factor)
]

outer_cross_factor = (occluder_radius / 65 ) - .2
# Define the vertices for a cross shape
outer_cross_vertices = [
    (-occluder_radius, -occluder_radius / outer_cross_factor), (-occluder_radius, occluder_radius / outer_cross_factor),
    (-occluder_radius / outer_cross_factor, occluder_radius / outer_cross_factor), (-occluder_radius / outer_cross_factor, occluder_radius),
    (occluder_radius / outer_cross_factor, occluder_radius), (occluder_radius / outer_cross_factor, occluder_radius / outer_cross_factor),
    (occluder_radius, occluder_radius / outer_cross_factor), (occluder_radius, -occluder_radius / outer_cross_factor),
    (occluder_radius / outer_cross_factor, -occluder_radius / outer_cross_factor), (occluder_radius / outer_cross_factor, -occluder_radius),
    (-occluder_radius / outer_cross_factor, -occluder_radius), (-occluder_radius / outer_cross_factor, -occluder_radius / outer_cross_factor)
]

# Create the cross shape # NOT USED ANYMORE
occluder_cross = visual.ShapeStim(
    win,
    vertices=cross_vertices,
    # fillColor=np.array(config["occluder_color"], dtype=float),
    # lineColor=np.array(config["occluder_color"], dtype=float),
    fillColor=[.5, .25, -.5],
    lineColor=[.5, .25, -.5],
    pos=(0, 0),
    opacity=occluder_opacity #if occluder_type[:5] == "cross" else 0,
)

# Add opacity = .5 to make see-through
occluder = visual.Rect(
    win,
    width=occluder_radius,
    height=occluder_radius,
    # width=occluder_radius * 1.1 if occluder_type == "cross_smooth" else occluder_radius * 1.5,
    # height=occluder_radius * 1.1 if occluder_type == "cross_smooth" else occluder_radius * 1.5,
    # fillColor=np.array(config["occluder_color"], dtype=float),
    # lineColor=np.array(config["occluder_color"], dtype=float),
    fillColor=[-0.05, -0.51, -0.87],
    lineColor=[-0.05, -0.51, -0.87],
    pos=(0, 0),
    opacity=occluder_opacity #if occluder_type != "cross" else 0,
    # ori=45 if occluder_type == "cross_smooth" else 0
)



# Create the inner outline shape
inner_outline = visual.ShapeStim(
    win,
    vertices=outer_cross_vertices,
    fillColor=[-0.5, -0.5, -0.5],  # Set fill color to dark grey using RGB values
    lineColor=[-0.5, -0.5, -0.5],  # Set line color to dark grey using RGB values
    pos=(0, 0),
    opacity=occluder_opacity,
)

occluder_glass = visual.Rect(
    win,
    width=2 * occluder_radius,
    height=2 * occluder_radius,
    fillColor="grey",
    lineColor="grey",
    pos=(0, 0),
    opacity=.5,
)

### Create borders to maintain square task screen

# Create the grey borders
left_border = visual.Rect(
    win=win,
    width=(win_dims[0] - square_size) / 2,
    height=win_dims[1],
    fillColor="black",
    lineColor="black",
    pos=[-(win_dims[0] - square_size) / 4 - square_size / 2, 0],
)

right_border = visual.Rect(
    win=win,
    width=(win_dims[0] - square_size) / 2,
    height=win_dims[1],
    fillColor="black",
    lineColor="black",
    pos=[(win_dims[0] - square_size) / 4 + square_size / 2, 0],
)

top_border = visual.Rect(
    win=win,
    width=win_dims[0],
    height=(win_dims[1] - square_size) / 2,
    fillColor="black",
    lineColor="black",
    pos=[0, (win_dims[1] - square_size) / 4 + square_size / 2],
)

bottom_border = visual.Rect(
    win=win,
    width=win_dims[0],
    height=(win_dims[1] - square_size) / 2,
    fillColor="black",
    lineColor="black",
    pos=[0, -(win_dims[1] - square_size) / 4 - square_size / 2],
)

##### DRAW GRID TO ALIGN INTERACTOR WITH
# Define the grid lines
line_length = 800  # Length of the lines to cover the window
line_width = 1  # Width of the lines
num_lines = 10  # Number of lines on each side of the center

# Create horizontal lines
horizontal_lines = []
for i in range(-num_lines, num_lines + 1):
    y = i * (line_length / (2 * num_lines))
    horizontal_lines.append(visual.Line(win, start=(-line_length / 2, y), end=(line_length / 2, y), lineWidth=line_width))

# Create vertical lines
vertical_lines = []
for i in range(-num_lines, num_lines + 1):
    x = i * (line_length / (2 * num_lines))
    vertical_lines.append(visual.Line(win, start=(x, -line_length / 2), end=(x, line_length / 2), lineWidth=line_width))

# Define the length and thickness of the cross arms
cross_length = fixation_length
cross_thickness = fixation_thickness

# Create the horizontal line of the cross
horizontal_line = visual.ShapeStim(
    win,
    vertices=[(-cross_length / 2, 0), (cross_length / 2, 0)],
    lineWidth=cross_thickness,
    closeShape=False,
    lineColor=fixation_color
)

# Create the vertical line of the cross
vertical_line = visual.ShapeStim(
    win,
    vertices=[(0, -cross_length / 2), (0, cross_length / 2)],
    lineWidth=cross_thickness,
    closeShape=False,
    lineColor=fixation_color
)

# Draw the fixation cross by drawing both lines
def draw_fixation():
    horizontal_line.draw()
    vertical_line.draw()

# Helper function to draw screen borders and other elements
def draw_screen_elements(trial, draw_occluder=False, draw_grid=False):
    left_border.draw()
    right_border.draw()
    top_border.draw()
    bottom_border.draw()
    
    # Draw interactor line if applicable
    if trial:
        if trial[:-2] == "45_top":
            line_135_top.draw()
        elif trial[:-2] == "45_bottom":
            line_135_bottom.draw()
        elif trial[:-2] == "135_top":
            line_45_top.draw()
        elif trial[:-2] == "135_bottom":
            line_45_bottom.draw()
    
    # Draw grid if enabled
    if draw_grid:
        for line in horizontal_lines + vertical_lines:
            line.draw()
            
    # Draw occluder if needed
    if draw_occluder:
        occluder.draw()
        
    # fixation.draw()
    draw_fixation()
