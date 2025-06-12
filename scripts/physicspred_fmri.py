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

#%%

# os.chdir("/Users/wieger.scheurer/exp_venv/lib/python3.10/site-packages/exptools2/experiments/physicspred/")


# Add repository paths - update these as needed
# sys.path.append("/Users/wiegerscheurer/repos/physicspred")
sys.path.append("/Users/wieger.scheurer/exp_venv/lib/python3.10/site-packages/exptools2")
sys.path.append("/Users/wieger.scheurer/exp_venv/lib/python3.10/site-packages/exptools2/experiments/physicspred/")
# sys.path.insert(0, '/Users/wieger.scheurer/miniconda3/envs/exp/lib/python3.10/site-packages/exptools2')
# sys.path.insert(0, '/Users/wieger.scheurer/miniconda3/envs/exp/lib/python3.10/site-packages/exptools2/experiments/physicspred')

# sys.path.append("D:/Users/wiesch/physicspred_cub6")

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

class BallTrial(Trial):
    """Trial for ball movement and hue change paradigm"""
    
    def __init__(self, session, trial_nr, phase_durations, trial_params, **kwargs):
        super().__init__(session, trial_nr, phase_durations, **kwargs)
        
        # Store trial parameters
        self.config = session.config
        self.win = session.win
        
        # Extract trial-specific parameters
        trial_idx = self.trial_nr % len(trial_params['trials'])
        self.trial = trial_params['trials'][trial_idx]
        self.bounce = trial_params['bounces'][trial_idx]
        self.ball_change = trial_params['ball_changes'][trial_idx]
        self.ball_color_change = trial_params['ball_color_changes'][trial_idx]
        self.ball_speed = trial_params['ball_speeds'][trial_idx]
        self.ball_start_color = trial_params['ball_start_colors'][trial_idx]
        self.rand_bounce_direction = trial_params['rand_bounce_directions'][trial_idx]
        
        # Calculate derived parameters
        self.changed_ball_color = oklab_to_rgb([(self.ball_start_color + self.ball_color_change), 0, 0], psychopy_rgb=True)
        
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
        self.movement_clock = None
    
    def prepare_trial(self):
        """Prepare the trial - called at the beginning of the trial"""
        # Initialize ball position and color
        # self.session.ball.pos = np.array(self.start_positions[self.edge])
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
                self.movement_clock = self.session.timer
                self.ballmov_start = self.movement_clock.getTime()
            # Keep running this phase until the time is up
            # while self.movement_clock.getTime() < (p3onset + self.phase_durations[self.phase]):
            # while (self.movement_clock - p3onset) < self.phase_durations[self.phase]: ### Figure out difference between timer and clock
            while self.session.timer.getTime() < 0:
                # Check for quit
                keys = event.getKeys(keyList=["escape", "q"])
                if "escape" in keys or "q" in keys:
                    break
                
                # self.session.ball.pos = self.start_positions[]
                # Draw current frame (used to be after process ball mov)
                
                self.session.ball.pos = np.array(self.start_positions[self.edge]) if not started_rolling else self.session.ball.pos
                self.session.ball.draw()


                # Process ball movement
                self.process_ball_movement()
                started_rolling = True

                self.draw_screen_elements(self.trial, draw_occluder=True)
                self.win.flip()

            
            # Reset movement clock for next trial
            self.movement_clock = None
        elif self.phase == 4: # Inter trial interval, perhaps this should be phase 0, but doesn't matter
            self.draw_screen_elements(None)

    
    def process_ball_movement(self):
        """Process ball movement for a single frame"""
        # square_size = self.config.display.square_size
        occluder_radius = self.config.occluder.radius
        ball_radius = self.config.ball.radius
        
        # Apply decay to velocity
        decay_factor = calculate_decay_factor(
            self.ball_speed, 
            # self.movement_clock.getTime(), 
            self.phase_durations[3] - self.session.timer.getTime(),  # Use session timer for consistent timing
            # self.session.timer.getTime(),  # Use session timer for consistent timing
            self.phase_durations[3],  # Ball movement duration
            constant=self.config.ball.decay_constant
        )
        
        self.velocity = [self.velocity[0] * decay_factor, self.velocity[1] * decay_factor]
        # self.velocity = [self.velocity[0] * decay_factor, self.velocity[1] * decay_factor]
        
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
                self.bounce_moment = self.movement_clock.getTime()
                self.bounce = False
                self.crossed_fixation = True
        
        # Check if ball is leaving occluder
        if (np.linalg.norm(self.session.ball.pos) > (occluder_radius / 2) - (ball_radius * 2)
            and self.crossed_fixation and not self.left_occluder):
            self.occluder_exit_moment = self.movement_clock.getTime()
            self.left_occluder = True
            print(f"Ball left occluder at {self.occluder_exit_moment - self.ballmov_start} (i.e. target onset in behav)")
            # WHY IS THIS NOT PRINTED IN SOME TRIALS? Presumably ones without an interactor
            # Also try to rewrite it in such way that it's the absolute value, from ball appearance
            # Now it's a negative, remainder
        
        # Handle ball color changes
        if (self.ball_change and self.crossed_fixation and 
            self.left_occluder and not self.hue_changed):
            # Set the change moment exactly once
            if self.ball_change_moment is None:
                self.ball_change_moment = self.movement_clock.getTime()

            # Apply the color change
            self.session.ball.color = self.changed_ball_color
            self.hue_changed = True
    
    def handle_normal_bounce(self):
        """Handle normal bounce physics when ball crosses the interactor line"""
        pre_bounce_velocity = self.pre_bounce_velocity
        pre_bounce_velocity = np.max(np.abs(self.velocity)) if pre_bounce_velocity is None else pre_bounce_velocity
        
        if self.trial[:2] == "45":
            velocity = collide(_flip_dir(self.edge), 45, pre_bounce_velocity)
            # velocity = collide((self.edge), 45, pre_bounce_velocity)
            bounce_moment = self.movement_clock.getTime()
        elif self.trial[:3] == "135":
            velocity = collide(_flip_dir(self.edge), 135, pre_bounce_velocity)
            # velocity = collide((self.edge), 135, pre_bounce_velocity)
            bounce_moment = self.movement_clock.getTime()
        
        return velocity, bounce_moment, pre_bounce_velocity
    
    def handle_phantom_bounce(self):
        """Handle phantom bounce physics for trials with no interactor"""
        pre_bounce_velocity = self.pre_bounce_velocity
        pre_bounce_velocity = np.max(np.abs(self.velocity)) if pre_bounce_velocity is None else pre_bounce_velocity
        
        if self.rand_bounce_direction == "left":
            velocity = _dir_to_velocity(
                # _rotate_90(_flip_dir(self.edge), "left"), pre_bounce_velocity
                _rotate_90((self.edge), "left"), pre_bounce_velocity
            )
        elif self.rand_bounce_direction == "right":
            velocity = _dir_to_velocity(
                # _rotate_90(_flip_dir(self.edge), "right"), pre_bounce_velocity
                _rotate_90((self.edge), "right"), pre_bounce_velocity
            )
        
        bounce_moment = self.movement_clock.getTime()
        return velocity, bounce_moment, pre_bounce_velocity, True
    
    def draw_screen_elements(self, trial_type, draw_occluder=False):
    # def draw(self, trial_type, draw_occluder=False):
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
        self.trial_params = self.create_design()
        
        # Hide mouse cursor
        self.win.mouseVisible = False
    
    def setup_visual_elements(self):
        """Set up all visual elements needed for the experiment"""
        config = self.config
        
        # Create ball
        self.ball = visual.Circle(
            win=self.win,
            radius=config.ball.radius,
            edges=32,
            fillColor=[1, 1, 1],
            lineColor=None,
            units='pix'
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
            units='pix'
        )
        
        self.right_border = visual.Rect(
            win=self.win,
            width=border_width,
            height=config.display.win_dims[1],
            fillColor='black',
            pos=(square_size/2 + border_width/2, 0),
            units='pix'
        )
        
        self.top_border = visual.Rect(
            win=self.win,
            width=config.display.win_dims[0],
            height=border_width,
            fillColor='black',
            pos=(0, square_size/2 + border_width/2),
            units='pix'
        )
        
        self.bottom_border = visual.Rect(
            win=self.win,
            width=config.display.win_dims[0],
            height=border_width,
            fillColor='black',
            pos=(0, -square_size/2 - border_width/2),
            units='pix'
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

    def create_design(self):
        """Create experiment design matrix and extract trial parameters"""
        config = self.config
        verbose = config.experiment.verbose
        n_trials = config.experiment.n_trials
        
        # Create experiment design matrix
        design_matrix = build_design_matrix(
            n_trials=n_trials,
            change_ratio=[True, False],
            ball_color_change_mean=config.ball.color_change_mean,
            # ball_color_change_mean=-10, # Debug for ball colour change
            ball_color_change_sd=config.ball.color_change_sd,
            verbose=verbose,
            neg_bias_factor=config.ball.neg_bias_factor,
            neg_bias_shift=config.ball.neg_bias_shift,
        )
        
        check_balance(design_matrix)

        # Extract trial parameters from design matrix
        trial_types = list(design_matrix["trial_type"])
        trials = list(design_matrix["trial_option"])
        bounces = list(design_matrix["bounce"])
        rand_bounce_directions = list(design_matrix["phant_bounce_direction"])
        ball_changes = list(design_matrix["ball_change"])
        ball_color_changes = list(design_matrix["ball_luminance"])
        
        # Generate ball speeds and starting colors
        ball_speeds = bellshape_sample(float(config.ball.avg_speed), float(config.ball.natural_speed_variance), n_trials)
        ball_start_colors = bellshape_sample(float(config.ball.start_color_mean), float(config.ball.start_color_sd), n_trials)
        
        # Generate inter-trial intervals, but now already done in the trial setup func, log there
        itis = truncated_exponential_decay(config.timing.min_iti, config.timing.max_iti, n_trials)
        
        # Create trial parameters dictionary
        trial_params = {
            "design_matrix": design_matrix,
            "trial_types": trial_types,
            "trials": trials,
            "bounces": bounces,
            "rand_bounce_directions": rand_bounce_directions,
            "ball_changes": ball_changes,
            "ball_color_changes": ball_color_changes,
            "ball_speeds": ball_speeds,
            "ball_start_colors": ball_start_colors,
            "itis": itis
        }
        
        return trial_params
        
    def create_trials(self, n_trials=None):
        """Create trials for the experiment"""
        from functions.utilities import truncated_exponential_decay

        if n_trials is None:
            n_trials = self.config.experiment.n_trials

        itis = truncated_exponential_decay(min_iti=self.config.timing.min_iti,
                                           truncation_cutoff=self.config.timing.max_iti,
                                           size=n_trials)
        
        # Create trials
        self.trials = []
        for trial_nr in range(n_trials):


            # Define phase durations
            phase_durations = [
                self.config.timing.fixation_dur,      # Fixation phase
                self.config.timing.interactor_dur,    # Interactor line display
                self.config.timing.occluder_dur,      # Occluder display
                self.config.timing.ballmov_dur,        # Ball movement
                itis[trial_nr - 1]                      # Inter Trial Interval 
            ]

            self.trials.append(
                BallTrial(
                    session=self,
                    trial_nr=trial_nr,
                    phase_durations=phase_durations,
                    trial_params=self.trial_params,
                    verbose=True,
                    timing='seconds'
                )
            )
            self.trials[trial_nr].prepare_trial()  # Prepare each trial # NOT SURE IF RIGHT PLACE

    def run(self):
        """Run the experiment"""
        # Start experiment
        self.start_experiment()
        
        # Display instructions (simplified)
        instructions = visual.TextStim(
            self.win,
            text="Ball Hue Experiment Demo\n\nPress 'Space' to start",
            color="white",
            pos=(0, 0),
            height=30
        )
        
        instructions.draw()
        self.win.flip()
        event.waitKeys(keyList=["space"])
        
        # Run all trials
        for trial_idx, trial in enumerate(self.trials):
            # Check for quit
            keys = event.getKeys(keyList=["escape", "q"])
            if "escape" in keys or "q" in keys:
                break
            
            print(f"Trial: {trial.trial}")
            trial.run()
        
        # Close experiment
        self.close()

if __name__ == "__main__":
    settings = os.path.join(os.path.dirname(__file__), os.pardir, "behav_settings.yml")

    # Create and run the session
    session = BallHueSession(output_str="sub-tosti", config_file=settings, settings_file=settings)
    session.create_trials(n_trials=session.config.experiment.n_trials)  # Reduce number of trials for testing
    session.run()
