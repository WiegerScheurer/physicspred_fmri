from psychopy import visual, core, event
import random
import time

# --- Set up the experiment window and load the videos ---
win = visual.Window([640, 480], color='black', units='pix', fullscr=True, screen=1)

print("Loading videos into memory...")
t0 = time.time() # Start the timer

video1_path = '/Users/wieger.scheurer/Pictures/hert_hapt.mp4'
video2_path = '/Users/wieger.scheurer/Pictures/beekje.mp4'

# Load the first video (for green trials)
video1 = visual.MovieStim(win, video1_path, size=(640, 480), flipVert=False, flipHoriz=False, autoStart=False)
# Load the second video (for red trials)
video2 = visual.MovieStim(win, video2_path, size=(640, 480), flipVert=False, flipHoriz=False, autoStart=False)

print(f"Videos loaded in {time.time() - t0:.2f} seconds.")

# --- Prepare the list of trials ---
num_trials = 10  # Total number of trials
# Make a list: half 'greentrial', half 'redtrial'
trial_types = ['greentrial'] * (num_trials // 2) + ['redtrial'] * (num_trials // 2)
# Randomly shuffle the order of the trials
random.shuffle(trial_types)
print(f"Trial order: {trial_types}")

# --- Run through all the trials one by one ---
video_durations = []  # This list will store how long each video was shown
framerate = win.getActualFrameRate()
if framerate is None:
    framerate = 60  # Fallback to 120Hz if undetectable
    print("WARNING: Could not detect actual frame rate. Assuming 60Hz.")

for i, trial in enumerate(trial_types):
    print(f"\nStarting trial {i+1}/{num_trials}: {trial}")

    # Choose which video to play for this trial
    if trial == 'greentrial':
        movie = video1  # Use the green trial video
    else:
        movie = video2  # Use the red trial video

    # Make sure the video starts from the beginning
    movie.seek(0)
    movie.play()  # Start playing the video

    # Start a timer to measure exactly how long the video is shown
    trial_clock = core.Clock()
    video_start = time.time()  # Record the wall clock time at the start

    # Show the video for exactly 1.5 seconds
    while trial_clock.getTime() < (1.5 - 1 / framerate):
        movie.draw()   # Tell PsychoPy to draw the video frame
        win.flip()     # Actually show the frame on the screen

        # Check if the user pressed 'escape' to quit early
        if event.getKeys(['escape']):
            print("Experiment aborted by user.")
            win.close()
            core.quit()

    movie.pause()  # Stop the video from playing any further
    win.flip()     # Clear the screen (show a blank frame)

    # Calculate how long the video was actually shown
    video_duration = time.time() - video_start
    video_durations.append(video_duration)  # Save this duration

    print(f"Trial {i+1} finished. Video duration: {video_duration:.4f} seconds.")

# --- After all trials are done, print a summary ---
print("\nAll trials complete.")
print("Video durations for each trial:")
for idx, dur in enumerate(video_durations):
    print(f"  Trial {idx+1}: {dur:.4f} seconds")

win.close()  # Close the experiment window