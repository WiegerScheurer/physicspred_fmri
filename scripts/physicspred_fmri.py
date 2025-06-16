########BACK UP BACK UP BACK UP BACK UP BACK UP BACK UP BACK UP BACK UP BACK UP BACK UP BACK UP BACK UP BACK UP
# TODO: Create a catch trial prompt that occurs after 64 trials, of which 32 contain a ball brightness change
# , These do not necessarily have to be balanced per block, but rather over the entire experiment. Find a simple way
# For them to respond, this will be done with an fMRI button box, so likely similar to how I do it with the buttonbox
# in the bheavioural experiment, using some sort of serial thing. 
# Furthermore, I need to make sure that it works on a block basis, so that it has 22 trials per block, and that 4 of these
# compose a single run, of which there are 4 in total. Hence, there's a total of 88 trials per run, and 352 trials in total.
# After every 22 trials, there will be a break of 30 seconds, and after every 88 trials, there will be a break of 1 or 2 minutes. (TODO:DECIDE)
# Also think about how to implement the eyetracker, and whether I want to do this in a separate script or not. (should not be necessary, but look
# at examples from Tomas). 
# Furthermore, I need to implement some type of feedback, which is presented after the catch trials (all of them?, there's 64, so can also do every 4)
# Yes I like 4, because then the first one will already be somewhat informative, and the frequency is nice. This means there'll be 16 in total, and they 
# will be 5s long as well, so that means 80 seconds of feedback. Also implement reaction time into this. 

# TODO: MAIL JOSE ONCE TIMING IS SOMEWHAT CLEAR. I believe I need to have a predefined number of volumes to scan (depends on TRs)

# TODO part 2: figure out to what extent the fMRI implementation is already functional. The example script seems very simple, I think
# most of it is implemented in the core .py files from the exptools2 wrapper, but I need to make sure I initialise it correctly. 
# Important to schedule some lab bookings, so that I can test all of that. Could then also test the example script. 



#%%
import os
import sys
import numpy as np
import random
from psychopy import visual, core, event
from omegaconf import OmegaConf
import exptools2
from exptools2.core import Session, Trial
from exptools2.core import PylinkEyetrackerSession
import pandas as pd
import os.path as op
from psychopy.visual import TextStim
import argparse

#%%

# os.chdir("/Users/wieger.scheurer/exp_venv/lib/python3.10/site-packages/exptools2/experiments/physicspred/")
sys.path.append("/Users/wieger.scheurer/exp_venv/lib/python3.10/site-packages/exptools2")
sys.path.append("/Users/wieger.scheurer/exp_venv/lib/python3.10/site-packages/exptools2/experiments/physicspred/")

# Import custom functions
from functions.utilities import (
    setup_folders,
    build_design_matrix,
    bellshape_sample,
    oklab_to_rgb,
    truncated_exponential_decay,
    get_pos_and_dirs,
    check_balance,
)

from functions.physics import (
    check_collision,
    collide,
    velocity_to_direction,
    predict_ball_path,
    _flip_dir,
    _rotate_90,
    _dir_to_velocity,
    will_cross_fixation,
    calculate_decay_factor,
    get_bounce_dist,
)

# class InstructionTrial(Trial):
#     """ Simple trial with instruction text. """

#     def __init__(self, session, trial_nr, phase_durations=[np.inf],
#                  txt=None, keys=None, draw_each_frame=False, **kwargs):

#         super().__init__(session, trial_nr, phase_durations, draw_each_frame=draw_each_frame, **kwargs)

#         self.trial = "instruction"
#         self.phase_names = ["instruction"]

#     def draw(self):
#         # Display intro text for the first trial
#         self.session.display_text(
#             text="Ball Hue Experiment Demo\n\nPress 'Space' to start",
#             keys=["space"],
#             height=30,
#             color="white",
#             pos=(0, 0)
#         )
#         self.stop_trial()

class InstructionTrial(Trial):
    """ Simple trial with instruction text. """

    def __init__(self, session, trial_nr, phase_durations=[np.inf],
                 txt=None, keys=None, draw_each_frame=False, **kwargs):

        super().__init__(session, trial_nr, phase_durations, draw_each_frame=draw_each_frame, **kwargs)

        self.trial = "instruction"
        self.phase_names = ["instruction"]

        txt_height = self.session.settings['various'].get('text_height')
        txt_width = self.session.settings['various'].get('text_width')
        text_position_x = self.session.settings['various'].get('text_position_x')
        text_position_y = self.session.settings['various'].get('text_position_y')

        if txt is None:
            txt = '''Press any button to commence.'''

        self.text = TextStim(self.session.win, txt,
                             height=txt_height, 
                             wrapWidth=txt_width, 
                             pos=[text_position_x, text_position_y],
                            #  font='Songti SC',
                             alignText = 'center',
                             anchorHoriz = 'center',
                             anchorVert = 'center')
        self.text.setSize(txt_height)

        self.keys = keys

    def draw(self):
        self.text.draw()
        self.session.win.flip()

    def get_events(self):
        events = super().get_events()

        if self.keys is None:
            if events:
                self.stop_phase()
        else:
            for key, t in events:
                if key in self.keys:
                    self.stop_phase()



class BallTrial(Trial):
    """Trial for ball movement and hue change paradigm"""
    
    def __init__(self, session, trial_nr, phase_durations, trial_params, 
                 phase_names, draw_each_frame=False, txt=None, keys=None, run_no=None, **kwargs):
        super().__init__(session, trial_nr, phase_durations, phase_names, draw_each_frame=draw_each_frame, **kwargs)
        
        self.trial_idx = self.trial_nr # NOT NEEDED Adjust trial index to match design matrix indexing (account for instruction trial)

        # Store trial parameters
        self.config = session.config
        # self.win = session.win

        # Extract trial-specific parameters
        self.trial_idx = self.trial_nr % len(session.dmx['trial_option']) # What does this do
        # if self.trial_idx > 0:

        self.trial = session.dmx['trial_option'][self.trial_idx]
        self.bounce = session.dmx['bounce'][self.trial_idx]
        self.ball_change = session.dmx['ball_change'][self.trial_idx]
        self.ball_color_change = session.dmx['ball_luminance'][self.trial_idx]
        self.ball_speed = session.dmx['ball_speeds'][self.trial_idx]
        self.ball_start_color = session.dmx['ball_start_colors'][self.trial_idx]
        self.rand_bounce_direction = session.dmx['phant_bounce_direction'][self.trial_idx]

        # Calculate derived parameters
        self.changed_ball_color = oklab_to_rgb([(self.ball_start_color + self.ball_color_change), 0, 0], psychopy_rgb=True)
        
        # Preparatory work for task prompt versions
        txt_height = self.session.settings['various'].get('text_height')
        txt_width = self.session.settings['various'].get('text_width')
        text_position_x = self.session.settings['various'].get('text_position_x')
        text_position_y = self.session.settings['various'].get('text_position_y')

        if txt is None:
            txt = '''Press any button to continue.'''

        txt = f"Did the ball change brightness?\nPress {self.session.button_map['no']} for NO or {self.session.button_map['yes']} for YES"
        self.keys = [self.session.button_map["no"], self.session.button_map["yes"]] 
        # self.response_given = False # This is to check if the response has been given, so that we can stop the phase

        self.text = TextStim(session.win, txt,
                            height=txt_height, 
                            wrapWidth=txt_width, 
                            pos=[text_position_x, text_position_y],
                            # font='Songti SC',
                            alignText = 'center',
                            anchorHoriz = 'center',
                            anchorVert = 'center')
        self.text.setSize(txt_height)


        # Extract start position letter from trial name
        if self.trial[:4] == "none":
            edge_letter = self.trial[-1]
        else:
            edge_letter = self.trial.split("_")[2]
            
        # Find the full edge option string
        edge_options = ["up", "down", "left", "right"]
        self.edge = _flip_dir(next(option for option in edge_options if option.startswith(edge_letter)))
            
        self.start_positions, self.directions = get_pos_and_dirs(
            ball_speed = self.ball_speed, # actual ball start speedo
            square_size=self.config.display.square_size, 
            ball_radius=self.config.ball.radius)

        # Initialize trial state variables
        self.velocity = None # maybe just set to actual start speed?
        self.bounce_moment = None
        self.pre_bounce_velocity = None
        self.crossed_fixation = False
        self.left_occluder = False
        self.hue_changed = False
        self.bounced_phantomly = False
        self.ball_change_moment = None
        self.occluder_exit_moment = None
        
        # Movement clock for tracking timing within phases
        self.movement_clock = None # I wonder if this is the right place, is it called every trial? or only during __init__
    
    def prepare_trial(self):
        """Prepare the trial - called at the beginning of the trial"""
        # Initialize ball position and color
        self.session.ball.color = np.clip(oklab_to_rgb([self.ball_start_color, 0, 0], psychopy_rgb=True), -1, 1) # Reset ball to start color of this trial
        
        # Set initial velocity
        self.velocity = np.array(self.directions[self.edge])
        
        # Reset trial state variables
        self.crossed_fixation = False
        self.left_occluder = False
        self.hue_changed = False
        self.bounce_moment = None
        self.pre_bounce_velocity = None
        self.ball_change_moment = None
        self.occluder_exit_moment = None
        self.response_given = False  # Reset response given state for each trial
        self.response_clock = None  # Reset response clock for each trial
        self.break_clock = None  # Reset break clock for each trial
    
    def draw(self):
        """Run a single phase of the trial"""

        started_rolling = False
        self.session.ball.color = np.clip(oklab_to_rgb([self.ball_start_color, 0, 0], psychopy_rgb=True), -1, 1)
        if self.phase == 0:  # Fixation phase
            self.draw_screen_elements(None)
        
        elif self.phase == 1:  # Interactor line display
            self.draw_screen_elements(self.trial)
        
        elif self.phase == 2:  # Occluder display
            self.draw_screen_elements(self.trial, draw_occluder=True)
            
        elif self.phase == 3:  # Ball movement phase
            if self.movement_clock is None:
                self.movement_clock = core.Clock()  # Create a new clock for this phase # NOT SURE IF THIS IS OK???
                self.movement_clock.reset()  # Start from 0
                self.ballmov_start = 0
                self.started_rolling = False
                # Set initial ball position
                self.session.ball.pos = np.array(self.start_positions[self.edge])
            
            # Process ball movement for this frame
            self.process_ball_movement()
            self.started_rolling = True
            
            # Draw everything
            self.session.ball.draw()
            self.draw_screen_elements(self.trial, draw_occluder=True)

        elif self.phase == 4: # Inter trial interval, perhaps this should be phase 0, but doesn't matter
            self.draw_screen_elements(None)

        elif self.phase == 5:  # Task prompt phase
            if self.phase_names[self.phase] == "task_prompt":
                if not self.response_given:  # Only draw if response has not been given yet
                    self.text.draw()
                    self.session.win.flip()  # Flip the window to show the text
                    self.draw_screen_elements(None)
                else:
                    # self.session.win.flip() #EDITED, see if this causes the fixation cross to be more dim (very weird)
                    self.draw_screen_elements(None)
                # Get start time of phase 5
                if self.response_clock is None:
                    self.response_clock = core.Clock()
                    self.response_clock.reset()  # Reset the clock for response timing

                
                time_to_respond = self.phase_durations[self.phase] - self.response_clock.getTime() # Get time since movement started + ITI duration

                if not hasattr(self, "last_print_time"):
                    self.last_print_time = None

                if self.last_print_time is None or abs(time_to_respond - self.last_print_time) >= 0.5:
                    print(f"Time to respond: {time_to_respond:.2f} seconds")
                    self.last_print_time = time_to_respond
            else:
                if self.break_clock is None:
                    self.break_clock = core.Clock()
                    self.break_clock.reset()  # Reset the clock for response timing

                time_to_break = self.phase_durations[self.phase] - self.break_clock.getTime() # Get time since movement started + ITI duration

                if not hasattr(self, "last_print_time"):
                    self.last_print_time = None

                if self.last_print_time is None or abs(time_to_break - self.last_print_time) >= 1:
                    self.last_print_time = time_to_break

                # self.session.display_text(text=f"Time for a break, chill it up in here!\n\nTask continues in {time_to_break:.2f} seconds",
                self.session.display_text(text=f"Time for a break, chill it up in here!\n\nTask continues in {int(time_to_break)} seconds",
                                        # keys=["space"],
                                        # duration=self.phase_durations[4],
                                        duration=1,
                                        # duration=10,  # for testing
                                        height=40,
                                        color="red",
                                        pos=(0, 350)
                                        )

        elif self.phase == 6:  # Post task prompt ITI
            self.draw_screen_elements(None)
        elif self.phase == 7:  # Short or long break (no conditional needed because you can only have 7 phases if there's a break)
            if self.break_clock is None:
                self.break_clock = core.Clock()
                self.break_clock.reset()  # Reset the clock for response timing
             # TODO: MAKE THIS MORE SMOOOTH, NOW IT'S LIKE A STROBOSOCOOP WHEN IT REFRESEHS
            time_to_break = self.phase_durations[self.phase] - self.break_clock.getTime() # Get time since movement started + ITI duration

            if not hasattr(self, "last_print_time"):
                self.last_print_time = None

            if self.last_print_time is None or abs(time_to_break - self.last_print_time) >= 1:
                self.last_print_time = time_to_break

            # self.session.display_text(text=f"Time for a break, chill it up in here!\n\nTask continues in {time_to_break:.2f} seconds",
            self.session.display_text(text=f"Time for a break, chill it up in here!\n\nTask continues in {int(time_to_break)} seconds",
                                    # keys=["space"],
                                    duration=self.phase_durations[7],
                                    # duration=1,
                                    # duration=10,  # for testing
                                    height=40,
                                    color="blue",
                                    pos=(0, 350)
                                    )
            
    def get_events(self):
        events = super().get_events()
        if self.phase == 5 and not self.response_given: # To make sure you can't always skip any phase by pressing a button
            if self.keys is None:
                if events:
                    self.response_given = True
                    # self.keys = None # to see if this can limit them to only a single response
            else:
                for key, t in events:
                    if key in self.keys:
                        self.response_given = True
                        # self.keys = None
        else: # EDITED perhaps does something but likely not
            pass
    
    def process_ball_movement(self):
        """Process ball movement for a single frame"""
        occluder_radius = self.config.occluder.radius
        ball_radius = self.config.ball.radius
        
        # Get elapsed time since ball movement started
        elapsed_time = self.movement_clock.getTime()
        
        # Apply decay to velocity
        decay_factor = calculate_decay_factor(
            self.ball_speed, 
            # elapsed_time,  # Time elapsed since movement started
            self.phase_durations[3] - elapsed_time,  # Time elapsed since movement started
            self.phase_durations[3],  # Total ball movement duration
            constant=self.config.ball.decay_constant
        )
        
        self.velocity = [self.velocity[0] * decay_factor, self.velocity[1] * decay_factor]
        
        # Update ball position
        self.session.ball.pos += tuple([self.velocity[0] * 1, self.velocity[1] * 1])  # Using skip_factor=1
        
        # Handle normal bounce
        if will_cross_fixation(self.session.ball.pos, self.velocity, 1) and self.bounce and self.trial[:4] != "none":
            self.velocity, self.bounce_moment, self.pre_bounce_velocity = self.handle_normal_bounce()
            self.bounce = False
            self.crossed_fixation = True
        
        # Handle phantom bounce or fixation crossing
        if will_cross_fixation(self.session.ball.pos, self.velocity, 1):
            if self.bounce and self.trial[:4] == "none":
                self.velocity, self.bounce_moment, self.pre_bounce_velocity, self.bounced_phantomly = self.handle_phantom_bounce()
            elif not self.bounce and not self.bounced_phantomly:
                self.bounce_moment = elapsed_time  # Use elapsed_time instead of movement_clock.getTime()
            self.bounce = False
            self.crossed_fixation = True
        
        # Check if ball is leaving occluder
        if (np.linalg.norm(self.session.ball.pos) > (occluder_radius / 2) - (ball_radius * 2)
            and self.crossed_fixation and not self.left_occluder):
            self.occluder_exit_moment = elapsed_time  # Use elapsed_time
            self.left_occluder = True
            print(f"Ball left occluder at {self.occluder_exit_moment:.3f} (i.e. target onset in behav)")

            idx = self.session.global_log.shape[0] # Check how large log currently is
            self.session.global_log.loc[idx - 1, 'occluder_exit'] = self.occluder_exit_moment

        # Handle ball color changes
        if (self.ball_change and self.crossed_fixation and 
            self.left_occluder and not self.hue_changed):
            # Set the change moment exactly once
            if self.ball_change_moment is None:
                self.ball_change_moment = elapsed_time  # Use elapsed_time

            # Apply the color change
            self.session.ball.color = self.changed_ball_color
            self.hue_changed = True


    def handle_normal_bounce(self):
        """Handle normal bounce physics when ball crosses the interactor line"""
        pre_bounce_velocity = self.pre_bounce_velocity
        pre_bounce_velocity = np.max(np.abs(self.velocity)) if pre_bounce_velocity is None else pre_bounce_velocity
        
        if self.trial[:2] == "45":
            velocity = collide(_flip_dir(self.edge), 45, pre_bounce_velocity)
            bounce_moment = self.movement_clock.getTime()  # This should now work correctly
        elif self.trial[:3] == "135":
            velocity = collide(_flip_dir(self.edge), 135, pre_bounce_velocity)
            bounce_moment = self.movement_clock.getTime()  # This should now work correctly
        
        return velocity, bounce_moment, pre_bounce_velocity

    def handle_phantom_bounce(self):
        """Handle phantom bounce physics for trials with no interactor"""
        pre_bounce_velocity = self.pre_bounce_velocity
        pre_bounce_velocity = np.max(np.abs(self.velocity)) if pre_bounce_velocity is None else pre_bounce_velocity
        
        if self.rand_bounce_direction == "left":
            velocity = _dir_to_velocity(
                _rotate_90((self.edge), "left"), pre_bounce_velocity
            )
        elif self.rand_bounce_direction == "right":
            velocity = _dir_to_velocity(
                _rotate_90((self.edge), "right"), pre_bounce_velocity
            )
        
        bounce_moment = self.movement_clock.getTime()  # This should now work correctly
        return velocity, bounce_moment, pre_bounce_velocity, True
    
    def draw_screen_elements(self, trial_type, draw_occluder=False):
        """Draw screen elements based on the current phase and trial type"""
        # Draw borders
        self.session.left_border.draw()
        self.session.right_border.draw()
        self.session.top_border.draw()
        self.session.bottom_border.draw()

        # Draw interactor lines based on trial type
        if trial_type is not None and trial_type[:4] != "none":
            if trial_type[:2] == "45":
                if "top" in trial_type:
                    self.session.line_45_top.draw()
                if "bottom" in trial_type:
                    self.session.line_45_bottom.draw()
            elif trial_type[:3] == "135":
                if "top" in trial_type:
                    self.session.line_135_top.draw()
                if "bottom" in trial_type:
                    self.session.line_135_bottom.draw()

        # Draw occluder if needed
        if draw_occluder:
            self.session.occluder.draw()
        
        # Draw fixation cross
        self.session.fixation_horizontal.draw()
        self.session.fixation_vertical.draw()

class BallHueSession(Session):
    """Session for the Ball Hue experiment"""
    
    def __init__(self, output_str, config_file="behav_settings.yml", output_dir=None, settings_file=None):
        super().__init__(output_str, output_dir=output_dir, settings_file=settings_file)
        
        # Load configuration file
        config_path = os.path.join(os.path.dirname(__file__), os.pardir, config_file)
        self.config = OmegaConf.load(config_path)
        
        # Setup visual elements
        self.setup_visual_elements()
        
        # Generate experiment design
        self.dmx = self.create_design()
        
        # Determine which button does what # Check if works
        button_options = ["left", "right"]
        button_order = random.sample(button_options, len(button_options))
        self.button_map = {
            "no": button_order[0], # Session because it's an input and the super class is not yet initialized
            "yes": button_order[1],
        }

        # Hide mouse cursor
        self.win.mouseVisible = False
    
    # def __init__(self, output_str, config_file="behav_settings.yml", output_dir=None, settings_file=None):
    #     super().__init__(output_str, output_dir=output_dir, settings_file=settings_file)
        
    #     # Load configuration file
    #     config_path = os.path.join(os.path.dirname(__file__), os.pardir, config_file)
    #     self.config = OmegaConf.load(config_path)
        
    #     # Setup visual elements
    #     self.setup_visual_elements()
        
    #     # Generate experiment design
    #     self.dmx = self.create_design()
        
    #     # Determine which button does what # Check if works
    #     button_options = ["left", "right"]
    #     button_order = random.sample(button_options, len(button_options))
    #     self.button_map = {
    #         "no": button_order[0], # Session because it's an input and the super class is not yet initialized
    #         "yes": button_order[1],
    #     }

    #     # Hide mouse cursor
    #     self.win.mouseVisible = False
    
    def setup_visual_elements(self):
        """Set up all visual elements needed for the experiment"""
        config = self.config
        
        # Create ball
        self.ball = visual.Circle(
            win=self.win,
            radius=config.ball.radius,
            edges=32,
            # fillColor=[1, 1, 1], # Perhaps change this? see if it matters?
            fillColor=[1, 1, .2], # Perhaps change this? see if it matters? doesnt appearas to be the case
            lineColor=None,
            units='pix',
            autoDraw=False, # THis does not solve anything, True just puts it on top of everything all the time,
        )
        
        # Create borders
        square_size = config.display.square_size
        border_width = (config.display.win_dims[0] - square_size) / 2 # config.display.border_width
        
        self.left_border = visual.Rect(
            win=self.win,
            width=border_width,
            height=config.display.win_dims[1],
            fillColor='black',
            pos=(-square_size/2 - border_width/2, 0),
            units='pix',
            autoDraw=True,  # Set to False to control when to draw it
        )
        
        self.right_border = visual.Rect(
            win=self.win,
            width=border_width,
            height=config.display.win_dims[1],
            fillColor='black',
            pos=(square_size/2 + border_width/2, 0),
            units='pix',
            autoDraw=True,  # Set to False to control when to draw it
        )
        
        self.top_border = visual.Rect(
            win=self.win,
            width=config.display.win_dims[0],
            height=border_width,
            fillColor='black',
            pos=(0, square_size/2 + border_width/2),
            units='pix',
            autoDraw=True,  # Set to False to control when to draw it
        )
        
        self.bottom_border = visual.Rect(
            win=self.win,
            width=config.display.win_dims[0],
            height=border_width,
            fillColor='black',
            pos=(0, -square_size/2 - border_width/2),
            units='pix',
            autoDraw=True,  # Set to False to control when to draw it
        )

        # from objects.task_components import (line_45_bottom, line_45_top, line_135_bottom, line_135_top)
        # self.line_45_top = line_45_top
        # self.line_45_bottom = line_45_bottom
        # self.line_135_top = line_135_top
        # self.line_135_bottom = line_135_bottom
        
        # Maybe this is aproblem? if not considered in pixel space
        bounce_dist = get_bounce_dist(config.ball.radius + (config.interactor.width / 2 * 1.8)) # 1.8 factor is due to the that now we use an image

        self.line_45_bottom = visual.ImageStim(
            self.win,
            image=config.interactor.path_45,
            size=(config.interactor.height, config.interactor.height),
            pos=(-bounce_dist, -(bounce_dist)),
            opacity=1,
            interpolate=True,
        )

        self.line_45_top = visual.ImageStim(
            self.win,
            image=config.interactor.path_45,
            size=(config.interactor.height, config.interactor.height),
            pos= ((bounce_dist), bounce_dist),
            opacity=1,
            interpolate=True,
        )

        self.line_135_bottom = visual.ImageStim(
            self.win,
            image=config.interactor.path_135,
            size=(config.interactor.height, config.interactor.height),
            pos=(bounce_dist, -(bounce_dist)),
            opacity=1,
            interpolate=True,
        )

        self.line_135_top = visual.ImageStim(
            self.win,
            image=config.interactor.path_135,
            size=(config.interactor.height, config.interactor.height),
            pos= (-(bounce_dist), bounce_dist),
            opacity=1,
            interpolate=True,
        )

        self.occluder = visual.Rect(
            self.win,
            width=config.occluder.radius,
            height=config.occluder.radius,
            fillColor=[-0.05, -0.51, -0.87],
            lineColor=[-0.05, -0.51, -0.87],
            pos=(0, 0),
            opacity=config.occluder.opacity,
            units='pix'
        )
            
        # Create the horizontal line of the cross
        self.fixation_horizontal = visual.ShapeStim(
            self.win,
            vertices=[(-config.fixation.length / 2, 0), (config.fixation.length / 2, 0)],
            lineWidth=config.fixation.thickness,
            closeShape=False,
            lineColor=config.fixation.color,
            units='pix'
        )

        # Create the vertical line of the cross
        self.fixation_vertical = visual.ShapeStim(
            self.win,
            vertices=[(0, -config.fixation.length / 2), (0, config.fixation.length / 2)],
            lineWidth=config.fixation.thickness,
            closeShape=False,
            lineColor=config.fixation.color,
            units='pix'
        )

    def create_design(self, save_design=True):
        """Create experiment design matrix and extract trial parameters"""
        config = self.config
        verbose = config.experiment.verbose
        fmri_trials = config.experiment.n_trials
        
        # Create experiment design matrix
        design_matrix = build_design_matrix(
            n_trials=fmri_trials,
            change_ratio=[True, False],
            ball_color_change_mean=config.ball.color_change_mean,
            ball_color_change_sd=config.ball.color_change_sd,
            verbose=verbose,
            neg_bias_factor=config.ball.neg_bias_factor,
            neg_bias_shift=config.ball.neg_bias_shift,
            fmri_task = True,
            prompt_every = 10, # Add 10% separately balanced task prompt trials
        )

        n_trials = len(design_matrix) # actual number of trials, including task prompts
        check_balance(design_matrix) # Remove eventually
        
        # Generate ball speeds and starting colors
        ball_speeds = bellshape_sample(float(config.ball.avg_speed), float(config.ball.natural_speed_variance), n_trials)
        ball_start_colors = bellshape_sample(float(config.ball.start_color_mean), float(config.ball.start_color_sd), n_trials)
        
        # Generate inter-trial intervals, but now already done in the trial setup func, log there
        itis = truncated_exponential_decay(config.timing.min_iti, config.timing.max_iti, n_trials)
        
        design_matrix["ball_speeds"] = ball_speeds
        design_matrix["ball_start_colors"] = ball_start_colors
        design_matrix["itis"] = itis

        # Save the rich design matrix to a TSV file
        design_matrix.to_csv(op.join(self.output_dir, f"{self.output_str}_design_matrix.tsv"), sep="\t", index=False)
        print(f"Design matrix saved to {self.output_dir} as TSV")

        return design_matrix # used to be trial_params
        
    def create_trials(self, n_trials=None, run_no:int=None):
        """Create trials for the experiment"""
        from functions.utilities import truncated_exponential_decay, trial_subset

        instruction_trial = InstructionTrial(session=self,
                                            # trial_nr=0,
                                            trial_nr=None,
                                            phase_durations=[np.inf],
                                            # txt=self.settings['stimuli'].get('instruction_text'),
                                            txt="Ball Hue Experiment Demo\n\nPress 'Space' to start",
                                            keys=['space'], 
                                            draw_each_frame=False)
        
        # itis = truncated_exponential_decay(min_iti=self.config.timing.min_iti,
        #                                    truncation_cutoff=self.config.timing.max_iti,
        #                                    size=n_trials)
        # Create trials
        self.trials = [instruction_trial]
        trial_counter = 1

        for trial_nr in range(n_trials):
        # TUrn into run-based selection
        # for trial_nr in range(start_trial, n_trials, )
            # Define phase durations
            phase_durations = [
                self.config.timing.fixation_dur,      # Fixation phase
                self.config.timing.interactor_dur,    # Interactor line display
                self.config.timing.occluder_dur,      # Occluder display
                self.config.timing.ballmov_dur,       # Ball movement
                # itis[trial_nr],                   # Inter Trial Interval (the 2 was a 1 first)
                self.dmx["itis"][trial_nr],                   # Inter Trial Interval (the 2 was a 1 first)
                # itis[trial_nr - 1],                   # Inter Trial Interval (the 2 was a 1 first)
            ]

            phase_names = ["fixation", "interactor", "occluder", "ball_movement", "iti"]

            # TODO: also add an extra, fixed (2s ?) ITI after prompt, to account for HRF decay
            # TODO: zorg ervoor dat prompt stopt na 3s (of langer als dat nodig is), zodat 't consistent is (maar chekc met micha en floris hoe belangrijk)
            if self.dmx["task_prompt"][trial_nr]:
                # Insert task_prompt duration & name before ITI (for consistency and HRF decay)
                phase_durations.extend([self.config.timing.task_prompt_dur, self.config.timing.post_prompt_iti])
                phase_names.extend(["task_prompt", "post_prompt_iti"])

            # if trial_nr % 22 == 0 and trial_nr > 0 and trial_nr % 88 != 0:
            if trial_nr % self.config.timing.short_break_freq == 0 and trial_nr > 0 and trial_nr % self.config.timing.long_break_freq != 0:
                # Add a break after every 22 trials
                phase_durations.append(self.config.timing.short_break_dur)
                phase_names.append("short break")
            elif trial_nr % self.config.timing.long_break_freq == 0 and trial_nr > 0:
                # Add a longer break after every 88 trials
                phase_durations.append(self.config.timing.long_break_dur)
                phase_names.append("long break")

            self.trials.append(
                BallTrial(
                    session=self,
                    trial_nr=(trial_nr),
                    # trial_nr=(trial_nr + 1),  # Start from 1 for trials, 0 is instruction trial
                    phase_durations=phase_durations,
                    phase_names=phase_names,
                    trial_params=self.dmx,
                    verbose=True,
                    timing='seconds',
                    draw_each_frame=True,  # setting to False makes nothing appear,
                    txt=None,  
                    keys=None,
                )
            )

            self.trials[trial_nr + 1].prepare_trial()  # Prep trial params

            # Increment trial counter (not used for anything as of now )
            trial_counter += 1

    def run(self):
        """Run the experiment"""
        # Start experiment
        self.start_experiment()

        # Run all trials
        for _, trial in enumerate(self.trials):
            print(f"Trial: {trial.trial_nr, trial.trial}")
            trial.run()
        
        # Close experiment
        self.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the fMRI ball bounce experiment in separate runs.')
    parser.add_argument('--subject', type=str, required=True, help='Subject identifier')
    parser.add_argument('--run', type=int, default=1, help='Run number for the experiment (default: 1)')

    args = parser.parse_args()

    runs_per_session = 4
    total_trials = 320

    config_path = os.path.join(os.path.dirname(__file__), os.pardir, "behav_settings.yml")

    # Create and run the session
    # session = BallHueSession(output_str="sub-potkwark", config_file=settings, settings_file=settings)
    session = BallHueSession(output_str=args.subject, config_file=config_path, run_no=args.run)
    
    session.create_trials(n_trials=len(session.dmx["trial_option"]), run_no=args.run)  # Reduce number of trials for testing

    session.run()
