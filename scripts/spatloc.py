# With the current setup, a single run of this script takes about 4.2min. 

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
'/Users/wieger.scheurer/exp_venv/lib/python3.10/site-packages/exptools2/experiments/physicspred'

# sys.path.append("/Users/wiegerscheurer/repos/physicspred")
sys.path.insert(0, '/Users/wieger.scheurer/miniconda3/envs/exp/lib/python3.10/site-packages/exptools2')
sys.path.insert(0, '/Users/wieger.scheurer/miniconda3/envs/exp/lib/python3.10/site-packages/exptools2/experiments/physicspred')

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
    design_spatloc,
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
        self.block = session.spatloc_blocks.iloc[trial_idx]
        
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
        self.session.ball.color = np.clip(oklab_to_rgb([self.ball_start_color, 0, 0], psychopy_rgb=True), -1, 1)
        
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
    
    # TODO: FIX THE TIMING OF THIS, FOR SOME REASON BALLS ONLY SHOW UP ON RANDOM TRIALS, fixed?
    # BUT THERE MUST BE SOME CONSISTENCY TO THIS, FIGURE OUT WHAT.
    def draw(self):
        """Run a single phase of the trial"""
        started_rolling = False
        spawn_range = None

        # Check for quit
        keys = event.getKeys(keyList=["escape", "q"])
        if "escape" in keys or "q" in keys:
            self.session.close()

        if self.phase == 0:  # Only occluder phase
            self.draw_screen_elements(self.trial, draw_occluder=True)

        elif self.phase == 1:  # Ball movement phase
            if self.movement_clock is None:
                self.movement_clock = self.session.timer
                self.ballmov_start = self.movement_clock.getTime()
            # Keep running this phase until the time is up
            while self.session.timer.getTime() < 0:

                self.session.ball.draw()
                self.session.ball.opacity = 1 # Make ball visible
    
                spawn_range, n_stutters, n_movs, stut_intervals, mov_variance = self.spatloc_stimbuild() if spawn_range is None else (spawn_range, n_stutters, n_movs, stut_intervals, mov_variance)

                print(f"Spawn range: {spawn_range}, n_stutters: {n_stutters}, n_movs: {n_movs}") if not started_rolling else None
                # Process ball movement
                self.spatloc_ballmov(location=self.block['location'], 
                                     movement=self.block['movement'], 
                                     phase_duration=self.phase_durations[1], 
                                     movement_duration=self.session.config.timing.movement_dur,
                                     spawn_range=spawn_range, 
                                     n_stutters=n_stutters,
                                     n_movs=n_movs,
                                     timer=self.session.timer,
                                     stut_intervals=stut_intervals,
                                     mov_variance=mov_variance)
                
                # self.process_ball_movement()
                started_rolling = True

                self.draw_screen_elements(self.trial, draw_occluder=True)
                self.win.flip()

            
            # Reset movement clock for next trial
            self.movement_clock = None

        elif self.phase == 2: # Inter trial interval, perhaps this should be phase 0, but doesn't matter
            self.draw_screen_elements(None, draw_occluder=False)

    def spatloc_stimbuild(self):
        """Initialize ball movement for a single frame"""
        
        # Calculate the number of events based on the phase duration and stutter interval
        n_stutters = int(self.phase_durations[1] // (self.session.config.timing.stutter_interval + self.session.config.timing.stutter_dur)) 
        n_movs = int(self.phase_durations[1] // (self.session.config.timing.movement_dur))

        # Calculate the inner and outer bounds for spawning, based on the occluder and ball radius
        inner_bound = self.session.config.occluder.radius // 2 + (self.session.config.ball.radius // 2)
        outer_bound = self.session.config.display.square_size // 2 - (self.session.config.ball.radius // 2)

        def generate_evenly_spread_separated_numbers(
            inner_bound, outer_bound, n_numbers,
            min_distance, subsequent_min_distance,
            jitter_within_segment=True,
            max_attempts_per_sample=1000
        ):
            segment_size = (outer_bound - inner_bound) / n_numbers
            numbers = []
            attempts = 0

            for i in range(n_numbers):
                segment_start = inner_bound + i * segment_size
                segment_end = segment_start + segment_size

                for attempt in range(max_attempts_per_sample):
                    if jitter_within_segment:
                        candidate = np.random.uniform(segment_start, segment_end)
                    else:
                        candidate = (segment_start + segment_end) / 2  # midden van segment
                    candidate = int(candidate)

                    # Check globale afstand
                    global_ok = all(abs(candidate - num) >= min_distance for num in numbers)
                    # Check lokale afstand
                    if numbers:
                        local_ok = abs(candidate - numbers[-1]) >= subsequent_min_distance
                    else:
                        local_ok = True  # eerste getal hoeft geen lokale check

                    if global_ok and local_ok:
                        numbers.append(candidate)
                        break  # succesvol
                else:
                    raise ValueError(
                        f"Failed to generate {n_numbers} separated numbers with min_distance={min_distance} "
                        f"and subsequent_min_distance={subsequent_min_distance} after {max_attempts_per_sample} attempts "
                        f"within segment {i}. Try relaxing constraints or increasing range."
                    )

            # Eventueel shufflen
            arr = np.array(numbers)
            np.random.shuffle(arr)
            return arr


        min_distance = 1
        subsequent_min_distance = 15 # 5 for bursts
        stut_per_loc = 3
        spawn_range_locs = generate_evenly_spread_separated_numbers(inner_bound, outer_bound, (n_stutters // stut_per_loc) + stut_per_loc, 
                                                      min_distance, subsequent_min_distance) # + margin of stutperloc

        spawn_range = np.repeat(spawn_range_locs, stut_per_loc)

        n = len(spawn_range)

        # Stutter interval variability
        stut_intervals = bellshape_sample(
            mean=self.session.config.timing.stutter_interval,
            sd=self.session.config.timing.stutter_interval * 0.1,  # 10% variability
            n_samples=n_stutters,
            shuffle=True,
        )
        mov_variance = bellshape_sample(
            mean=0, #self.session.config.timing.movement_dur,
            sd=self.session.config.timing.movement_dur * 0.25,  # 10% variability
            n_samples=n_movs,
            shuffle=True,
        )

        return spawn_range, n_stutters, n_movs, stut_intervals, mov_variance
    
    def spatloc_ballmov(
        self, location: str, movement: str, phase_duration: float,
        movement_duration: float, spawn_range, n_stutters: int, n_movs: int,timer, stut_intervals, mov_variance,
        ball_task=False, target_indices=None, target_luminance_change=-0.9
    ):
        """
        Handles both stutter and repeated in/out ball movement for a given phase.
        - 'stutter': Ball appears at positions for stutter_dur, then disappears for stutter_interval.
        - 'in'/'out': Ball moves smoothly in or out, repeating with variable durations based on mov_variance.
        
        Parameters:
        -----------
        ball_task : bool
            If True, introduces target balls with altered luminance
        target_indices : list or None
            Indices of events that should be targets (if None and ball_task=True, generates random targets)
        target_luminance_change : float
            Amount to change the luminance by for target balls (0-1 scale)
        """

        loc_map = {
            "left": (-1, 0),
            "right": (1, 0),
            "top": (0, 1),
            "bottom": (0, -1)
        }
        
        n_events = n_stutters if movement == "stutter" else n_movs # cleaner

        # Timing config for stutter
        stutter_dur = self.session.config.timing.stutter_dur
        
        # Store original ball color if we need to modify it for targets
        if not hasattr(self, '_original_ball_color'):
            # Store color in the format PsychoPy is expecting
            if hasattr(self.session.ball, 'fillColor'):
                self._original_ball_color = self.session.ball.fillColor
            else:
                self._original_ball_color = self.session.ball.color
        
        original_color = self._original_ball_color
            
        # Generate target indices if needed
        if ball_task and target_indices is None:
            # Generate random indices, approximately 20% of events
            num_targets = max(1, int(n_events * 0.2))
            target_indices = random.sample(range(n_events), num_targets)
        
        # Movement amplitude based on display config
        movement_amplitude = 0.51 * self.session.config.display.square_size + self.session.config.ball.radius
        
        # Current time: goes from -phase_duration to 0
        current_time = timer.getTime()
        
        # Pre-emptively hide the ball before checking if we're in phase
        # This ensures clean transitions between phases
        if current_time > -0.05 or current_time < -phase_duration:
            self.session.ball.opacity = 0  # Hide ball right at phase boundary
            return  # Outside of phase
        
        # Add a small buffer time at the end of phase to ensure clean transitions
        if current_time > -0.05:
            self.session.ball.opacity = 0
            return
        
        # Compute elapsed time since phase start
        elapsed = phase_duration + current_time  # goes from 0 to phase_duration
        
        if movement == "stutter":
            # Calculate optimally fitting stutter events with variable intervals
            stutter_timings = []
            start_time = 0
            
            for i in range(n_events):
                if i < len(stut_intervals):
                    interval = stut_intervals[i]
                else:
                    interval = self.session.config.stutter_interval  # Default if not enough provided
                    
                # Each event consists of a visible period followed by an interval
                stutter_timings.append({
                    'start': start_time,
                    'end': start_time + stutter_dur,
                    'index': i
                })
                start_time += stutter_dur + interval
                
                # Check if we've filled the phase duration
                if start_time > phase_duration - 0.05:
                    break
                    
            # Determine current stutter event
            current_stutter = None
            for stutter in stutter_timings:
                if stutter['start'] <= elapsed < stutter['end']:
                    current_stutter = stutter
                    break
                    
            # Handle ball visibility and position
            if current_stutter is not None:
                event_idx = current_stutter['index']
                self.session.ball.pos = tuple(np.array(loc_map[location]) * spawn_range[event_idx])
                self.session.ball.opacity = 1  # Make ball visible
                
                # Handle target ball if needed
                if ball_task and event_idx in target_indices:
                    # Modify ball color (increase/decrease luminance)
                    try:
                        # Check if we're dealing with a color name, RGB, or other format
                        if hasattr(self.session.ball, 'fillColor'):
                            # For PsychoPy visual objects that use fillColor
                            if self.session.ball.colorSpace == 'rgb':
                                # For RGB colorspace (-1 to 1 scale in PsychoPy)
                                self.session.ball.fillColor = [
                                    min(1.0, max(-1.0, self.session.ball.fillColor[i] + target_luminance_change))
                                    for i in range(len(self.session.ball.fillColor))
                                ]
                            else:
                                # For other colorspaces, just use a brightened version of original
                                self.session.ball.fillColor = original_color
                        else:
                            # For objects using standard color attribute
                            if hasattr(self.session.ball, 'colorSpace') and self.session.ball.colorSpace == 'rgb':
                                self.session.ball.color = [
                                    min(1.0, max(-1.0, self.session.ball.color[i] + target_luminance_change))
                                    for i in range(len(self.session.ball.color))
                                ]
                            else:
                                self.session.ball.color = original_color
                    except Exception as e:
                        # If any error occurs, fall back to original color
                        if hasattr(self.session.ball, 'fillColor'):
                            self.session.ball.fillColor = original_color
                        else:
                            self.session.ball.color = original_color
                else:
                    # Reset to original color
                    if hasattr(self.session.ball, 'fillColor'):
                        self.session.ball.fillColor = original_color
                    else:
                        self.session.ball.color = original_color
            else:
                self.session.ball.opacity = 0  # Hide ball between stutters
                
        elif movement in ("in", "out"):
            # Calculate optimal number of complete movements that fit in the phase
            # First calculate actual movement durations
            actual_durations = [movement_duration + var for var in mov_variance[:n_events]]
            
            # Determine how many complete movements we can fit within phase_duration
            # allowing for a small buffer at the end
            total_duration = sum(actual_durations[:n_events])
            buffer_time = 0.05  # 50ms buffer at the end of phase
            
            # If total duration is too long, reduce number of events
            if total_duration > phase_duration - buffer_time:
                # Find max number of complete movements that fit
                fit_duration = 0
                events_that_fit = 0
                for i in range(n_events):
                    if fit_duration + actual_durations[i] <= phase_duration - buffer_time:
                        fit_duration += actual_durations[i]
                        events_that_fit += 1
                    else:
                        break
                n_events = events_that_fit
            
            # If no movements fit, hide ball and return
            if n_events == 0:
                self.session.ball.opacity = 0
                return
            
            # Calculate start times for each movement
            movement_start_times = [0]
            for i in range(1, n_events):
                movement_start_times.append(movement_start_times[i-1] + actual_durations[i-1])
            
            # Calculate end time for the last movement
            end_time = movement_start_times[-1] + actual_durations[-1]
            
            # If we're past the end of all movements
            if elapsed >= end_time:
                self.session.ball.opacity = 0  # Hide ball
                return
            
            # Find which movement we're currently in
            current_movement = None
            for i in range(n_events):
                if i == n_events - 1:  # Last movement
                    if movement_start_times[i] <= elapsed < (movement_start_times[i] + actual_durations[i]):
                        current_movement = i
                        break
                else:
                    if movement_start_times[i] <= elapsed < movement_start_times[i+1]:
                        current_movement = i
                        break
            
            # If we're not in any movement, hide ball
            if current_movement is None:
                self.session.ball.opacity = 0
                return
            
            # Calculate progress within the current movement
            movement_progress = (elapsed - movement_start_times[current_movement]) / actual_durations[current_movement]
            movement_progress = np.clip(movement_progress, 0, 1)
            
            # Set position based on movement direction
            if movement == "in":
                pos_factor = 1 - movement_progress  # Edge to center
            else:  # "out"
                pos_factor = movement_progress  # Center to edge
                
            direction = np.array(loc_map[location])
            self.session.ball.pos = tuple(direction * pos_factor * movement_amplitude)
            self.session.ball.color = np.clip(oklab_to_rgb([self.ball_start_color, 0, 0], psychopy_rgb=True), -1, 1)

            self.session.ball.opacity = 1  # Make ball visible
            
            # Handle target ball if needed
            if ball_task and current_movement in target_indices:
                # Modify ball color for target
                try:
                    # Check if we're dealing with a color name, RGB, or other format
                    if hasattr(self.session.ball, 'fillColor'):
                        # For PsychoPy visual objects that use fillColor
                        if self.session.ball.colorSpace == 'rgb':
                            # For RGB colorspace (-1 to 1 scale in PsychoPy)
                            self.session.ball.fillColor = [
                                min(1.0, max(-1.0, self.session.ball.fillColor[i] + target_luminance_change))
                                for i in range(len(self.session.ball.fillColor))
                            ]
                        else:
                            # For other colorspaces, just use a brightened version of original
                            self.session.ball.fillColor = original_color
                    else:
                        # For objects using standard color attribute
                        if hasattr(self.session.ball, 'colorSpace') and self.session.ball.colorSpace == 'rgb':
                            self.session.ball.color = [
                                min(1.0, max(-1.0, self.session.ball.color[i] + target_luminance_change))
                                for i in range(len(self.session.ball.color))
                            ]
                        else:
                            self.session.ball.color = original_color
                except Exception as e:
                    # If any error occurs, fall back to original color
                    if hasattr(self.session.ball, 'fillColor'):
                        self.session.ball.fillColor = original_color
                    else:
                        self.session.ball.color = original_color
            else:
                # Reset to original color
                if hasattr(self.session.ball, 'fillColor'):
                    self.session.ball.fillColor = original_color
                else:
                    self.session.ball.color = original_color
        else:
            self.session.ball.opacity = 0  # Hide ball for unhandled movement types
            return
    
    def process_ball_movement(self):
        """Process ball movement for a single frame"""
        # square_size = self.config.display.square_size
        
        # Apply decay to velocity
        decay_factor = calculate_decay_factor(
            self.ball_speed, 
            # self.movement_clock.getTime(), 
            self.phase_durations[1] - self.session.timer.getTime(),  # Use session timer for consistent timing
            # self.session.timer.getTime(),  # Use session timer for consistent timing
            self.phase_durations[1],  # Ball movement duration
            constant=self.config.ball.decay_constant
        )
        
        self.velocity = [self.velocity[0] * decay_factor, self.velocity[1] * decay_factor]
        
        # Update ball position
        self.session.ball.pos += tuple([self.velocity[0] * 1, self.velocity[1] * 1])  # Using skip_factor=1
        
        # Handle ball color changes
        if (self.ball_change and self.crossed_fixation and 
            self.left_occluder and not self.hue_changed):
            # Set the change moment exactly once
            if self.ball_change_moment is None:
                self.ball_change_moment = self.movement_clock.getTime()

            # Apply the color change
            self.session.ball.color = self.changed_ball_color
            self.hue_changed = True
    
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
            self.session.occluder.setAutoDraw(True)

        # Draw fixation cross
        self.session.fixation_horizontal.draw()
        self.session.fixation_vertical.draw()
        self.session.fixation_horizontal.setAutoDraw(True)
        self.session.fixation_vertical.setAutoDraw(True)

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
        
        # Make the spatial localiser block design
        self.spatloc_blocks = design_spatloc()

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
            units='pix',
            # opacity=.5,
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
            units='pix',
            # opacity=.5,
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
        n_trials = config.experiment.n_trials_spatloc
        
        # Create experiment design matrix
        design_matrix = build_design_matrix(
            n_trials=n_trials,
            change_ratio=[True],
            ball_color_change_mean=config.ball.color_change_mean,
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
            n_trials = self.config.experiment.n_trials_spatloc

        itis = truncated_exponential_decay(min_iti=self.config.timing.min_iti,
                                           truncation_cutoff=self.config.timing.max_iti,
                                           size=n_trials)
        
        # Create trials
        self.trials = []
        for trial_nr in range(n_trials):

            # Define phase durations
            phase_durations = [
                self.config.timing.spatloc_occluder_dur,      # Only occluder
                self.config.timing.spatloc_ballmov_dur,    # Ball movement
                self.config.timing.spatloc_iti_dur,      # Time between trials
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

            print(f"Trial: {trial.trial_nr}, Location: {trial.block['location']}, Movement: {trial.block['movement']}")

            trial.run()
        
        # Close experiment
        self.close()

if __name__ == "__main__":
    settings = os.path.join(os.path.dirname(__file__), os.pardir, "behav_settings.yml")

    # Create and run the session
    session = BallHueSession(output_str="sub-test", config_file=settings, settings_file=settings)
    session.create_trials(n_trials=session.config.experiment.n_trials_spatloc)  # Reduce number of trials for testing
    session.run()

