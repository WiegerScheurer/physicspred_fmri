#!/usr/bin/env python

import os.path as op
import sys
# sys.path.insert(0, '/Users/wiegerscheurer/repos/exptools2/local_exptools2/')
# sys.path.insert(0, '/Users/wieger.scheurer/repoclones/physicspred_exp/local_exptools2/')
# print(sys.path)

# import local_exptools2

# from local_exptools2.core import Session
from exptools2.core import Session
# from local_exptools2.core import Trial
from exptools2.core import Trial
from psychopy.visual import TextStim
from psychopy import visual
from exptools2 import utils, stimuli
# from local_exptools2 import utils, stimuli
import os



class BehavTrial(Trial):
    """ Simple trial with text (trial x) and fixation. """
    def __init__(self, session, trial_nr, phase_durations, txt=None, trial_params=None, **kwargs):
        super().__init__(session, trial_nr, phase_durations, trial_params, **kwargs)
        self.txt = visual.TextStim(self.session.win, txt) 
        self.trial_params = trial_params

    def draw(self):
        """ Draws stimuli """
        [border.draw() for border in self.session.screen_borders] # Draw screen borders

        if self.phase == 0:
            self.txt.draw()
        else:
            self.session.occluder.draw() # Draw occluder
            [fix_line.draw() for fix_line in self.session.cross_fix] # Draw fixation cross
            if self.trial_params["trial_option"][:4] != "none": # Draw interactor if applicable
                stimuli.create_interactor(self.session.win, 
                                          self.trial_params["trial_option"],
                                          self.session.settings["ball"]["radius"],
                                          **self.session.settings["interactor"]).draw()
                

        
class BehavSession(Session):
    """Behaviour session for physics prediction experiment."""    
    
    def __init__(self, output_str, output_dir=None, settings_file=None, n_trials=10):
        """ Initializes BehavSession object. """
        self.n_trials = n_trials

        super().__init__(output_str, output_dir=output_dir, settings_file=settings_file)
        
        self.design_matrix = utils.build_design_matrix(
            n_trials = n_trials,
            change_ratio=[True],
            ball_color_change_mean=self.settings["ball"]["color_change_mean"], 
            ball_color_change_sd=self.settings["ball"]["color_change_sd"],
            verbose=self.settings["experiment"]["verbose"],
            neg_bias_factor=self.settings["ball"]["neg_bias_factor"],
        
        )

        
    def create_trials(self, durations=(.5, .5), timing='seconds'):
        self.trials = []
        for trial_nr in range(self.n_trials):
            
            self.trials.append(
                BehavTrial(session=self,
                          trial_nr=trial_nr,
                          phase_durations=durations,
                          txt='Trial numero %i' % trial_nr,
                          parameters=dict(trial_type='even' if trial_nr % 2 == 0 else 'odd'),
                          verbose=True,
                          timing=timing,
                          trial_params=self.design_matrix.iloc[trial_nr],)
            )

    def run(self):
        """ Runs experiment. """
        self.start_experiment()
        for trial in self.trials:
            trial.run()            

        self.close()
        
        
if __name__ == '__main__': # This is so that it doesn't run when imported as a module

    # settings = op.join(op.dirname(__file__), 'behav_settings.yml')
    settings = op.join(op.abspath(op.join(op.dirname(__file__), '..')), 'behav_settings.yml')
    session = BehavSession('sub-03', n_trials=20, output_dir=op.join(os.getcwd(), "logs/wip"),settings_file=settings)
    session.create_trials(durations=(1, .25), timing='seconds')
    #session.create_trials(durations=(3, 3), timing='frames')
    session.run()
    session.quit()
    
    
