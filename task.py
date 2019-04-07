import numpy as np
from physics_sim import PhysicsSim


class Task:
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(
        self,
        init_pose=None,
        init_velocities=None,
        init_angle_velocities=None,
        runtime=5.0,
        target_pos=None,
    ):
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
        self.sim = PhysicsSim(
            init_pose, init_velocities, init_angle_velocities, runtime
        )
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 300
        self.action_high = 700
        self.action_size = 4

        # Goal
        self.target_pos = (
            target_pos if target_pos is not None else np.array([0.0, 0.0, 10.0])
        )

    def _distance_to_target(self):
        return np.sqrt(np.square(self.sim.pose[:3] - self.target_pos).sum())

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # reward = 2 * (self.sim.pose[2])

        # Base reward function
        reward = 10.0 / (self._distance_to_target() / 2)

        # Punish flying low
        reward -= 10 - self.sim.pose[2]

        # # Reward being within 1 unit radius of target
        # if distance_to_target < 2:
        #     reward += 100

        # # Punish crashes

        # Punish Euler angles more than 30 degrees
        # for angle in np.sin(self.sim.pose[3:]):
        #     if abs(angle) > 0.5:
        #         reward -= 100

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(
                rotor_speeds
            )  # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)

        # Punish crashes and reward being in flight
        if done:
            if self._distance_to_target() < 15:
                reward += 100
            elif self.sim.pose[2] <= 0:
                reward -= 10

        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
