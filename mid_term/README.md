Game Rules ********************************************************************
Reward:
    -1  for every time step
    -10 for hazards
    -10 & push four cells down if agent touches the moving hazard (large game)
    +10 for reaching the final state
*******************************************************************************

NOTE:
1- you can run the display method (by default) to play with the keyboard.
You can use the arrow keys to move around, or press the space bar to watch the agent play its policy.

2- To pass in a trajectory, you can use the animate method (uncomment) and pass in a list of tuples as a trajectory.
If you call the animate method without specifying a trajectory, it will generate a trajectory from a random walk policy.
So BE CAREFUL, generating such a trajectory for the big map can take a long time! The agent would have to press the blue button, and then travel all the way to the target and avoid the moving hazard.
