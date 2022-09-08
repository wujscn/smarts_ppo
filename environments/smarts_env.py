
import secrets
import gym
import numpy as np
import glob
import random
import time

from smarts.env.hiway_env import HiWayEnv
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import NeighborhoodVehicles, RGB
from smarts.core.controllers import ActionSpaceType

class SmartsEnv():
    def __init__(self, scenario_path=[]):
        # TODO(wujs): make it convinient
        # self.ACTION_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.ACTION_SPACE = gym.spaces.Discrete(4)
        self.OBSERVATION_SPACE = gym.spaces.Box(low=0, high=1, shape=(80, 80, 9))
        self.AGENT_ID = 'Agent-sheng'
        self.states = np.zeros(shape=(80, 80, 9))

        # TODO(wujs): make global var to specific scenario
        # scenario_path = ['scenarios/left_turn']
        # scenario_path = ['scenarios/roundabout']
        scenario_path = ['scenarios/1_to_2lane_left_turn_t']
        max_episode_steps = 600

        # define agent interface
        agent_interface = AgentInterface(
            max_episode_steps=max_episode_steps,
            waypoints=True,
            neighborhood_vehicles=NeighborhoodVehicles(radius=60),
            rgb=RGB(80, 80, 32/80), 
            action=ActionSpaceType.Lane,
        )

        # define agent specs
        agent_spec = AgentSpec(
            interface=agent_interface,
            observation_adapter=self.observation_adapter,
            reward_adapter=self.reward_adapter,
            action_adapter=self.action_adapter,
            info_adapter=self.info_adapter,
        )
        # TODO(wujs): deal with seed, make seed random or something
        self._env = HiWayEnv(scenarios=scenario_path, 
                                agent_specs={self.AGENT_ID: agent_spec}, 
                                headless=True,
                                seed=1, 
                                visdom=True)
        self._env.observation_space = self.OBSERVATION_SPACE
        self._env.action_space = self.ACTION_SPACE
        self._env.agent_id = self.AGENT_ID

    @property
    def observation_space(self):
        return self._env.observation_space
    
    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):
        self._rewards = []
        obs = self._env.reset()
        return obs[self._env.agent_id]

    def step(self, action):
        obs, reward, done, info = self._env.step({self._env.agent_id: action})
        obs = obs[self._env.agent_id]
        reward = reward[self._env.agent_id]
        done = done[self._env.agent_id]
        info = info[self._env.agent_id]

        self._rewards.append(reward)
        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None
        # return obs * self._obs_mask, reward / 100.0, done, info
        return obs, reward / 100.0, done, info

    def render(self):
        self._env.render()
        time.sleep(0.033)

    def close(self):
        self._env.close()

    # *================================== Samrts ==================================
    # observation space
    def observation_adapter(self, env_obs):
        new_obs = env_obs.top_down_rgb[1] / 255.0
        self.states[:, :, 0:3] = self.states[:, :, 3:6]
        self.states[:, :, 3:6] = self.states[:, :, 6:9]
        self.states[:, :, 6:9] = new_obs

        if env_obs.events.collisions or env_obs.events.reached_goal:
            self.states = np.zeros(shape=(80, 80, 9))

        return np.array(self.states, dtype=np.float32)

    # reward function
    def reward_adapter(self, env_obs, env_reward):
        # progress = env_obs.ego_vehicle_state.speed * 0.1
        if env_obs.events.reached_goal:
            goal = 10
            print("Hooya! Reach the goal!") 
        else:
            goal = 0

        reward = goal # 0.01 * progress

        if env_obs.events.off_road:
            reward -= 1
            print("went off_road")

        if env_obs.events.collisions:
            reward -= 10
            print("CRASH!") 
        
        # if env_obs.events.on_shoulder:
        #     reward -= 1
        reward += env_reward + 0.001  # round_1:reward = reward +env_reward

        return reward

    # action space
    # def action_adapter(self, model_action): 
    #     speed = model_action[0] # output (-1, 1)
    #     speed = (speed - (-1)) * (10 - 0) / (1 - (-1)) # scale to (0, 10)
        
    #     speed = np.clip(speed, 0, 10)
    #     model_action[1] = np.clip(model_action[1], -1, 1)

    #     # discretization
    #     if model_action[1] < -1/3:
    #         lane = -1
    #     elif model_action[1] > 1/3:
    #         lane = 1
    #     else:
    #         lane = 0

    #     return (speed, lane)
    def action_adapter(self, model_action):
        action_map = ["keep_lane", "slow_down", "change_lane_left", "change_lane_right"]
        return action_map[model_action]

    # information
    def info_adapter(self, observation, reward, info):
        return info