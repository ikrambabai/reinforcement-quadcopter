import numpy as np
from physics_sim import PhysicsSim


class Task:
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)

        # Inspired by the 'original' DDGP paper, we use the concept of action_repeats (read the paper).
        self.action_repeat = 3

        # state can be more sophisticated than just the 6-dimensional pos - can tinker here
        # Also, state_size must take action_repeats into account
        self.state_size = self.action_repeat * 6

        # The environment has 4-dimensional action-space - each entry for one of the 4 rotors.
        # Each of these will have a defined min and max values.
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal (use the given target position, else use 0, 0, 10)
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """continue flying (rewarded) or going farther from the the target (penalized)"""
        reward = 1. - .003 * (abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        done = None
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
