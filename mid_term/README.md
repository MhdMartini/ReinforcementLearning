Please uncomment the necessary lines to run what you want. Eventually, I will parse command line arguemnts so it is
more user friendly, so sorry about the hassel for now.

"""Notes about the game"""

*Small World*
The agent must reach the final state which is surrounded by hazardous cells from all directions.
In order to reach the final state, the agent must press a button which vanishes all hazardous cells and makes the terminal state reachable.
Reward:
    -1  for every time step
    -10 for touching the hazardous zone
    +10 for reaching the final state

*Large World*
Initially, I intended the large world to be a larger copy of the small world, with perhaps multiple buttons or keys.
I constructed the large map, but I did not add any additional complexity than in the small world. This is partly because I did not have
enough time to work on it, but also because I am thinking of transitioning to pong, or something more exciting.

"""HOW TO RUN"""

You can uncomment lines 199 or 202 to use a small or large map, respectively.

You can uncomment lines 208 (the display method) to play with the keyboard, or watch the agent play its policy by presisng the space bar.
you can alternate between keyboard control and agent control by pressing the space bar.

Another method is provided to visualizing a specific trajectory (the animate method). To do this, uncomment line 214. If no
trajectory is specified, the agent will generate a trajectory based on its policy.
NOTE: a valid trajectory for the large map may take a while to generate.


"""Final notes"""
I apologize for the code structure not matching the template we were provided with, which might make it harder to integrate code
from different students'. I wrote most of this code before Tueseday, and did not have much time to improve it since then. It wasn't
clear to me that we had to follow the template - I thought it was there just for reference or general guidance.
