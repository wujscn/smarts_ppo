
from pyexpat import model
import secrets
import gym
import numpy as np
import glob
import random
import time

from smarts.env.hiway_env import HiWayEnv
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import NeighborhoodVehicles, RGB, EventConfiguration
from smarts.core.controllers import ActionSpaceType
from smarts import sstudio

class SmartsEnv():
    def __init__(self, scenario_path=[], envision=False, visdom=False, sumo=True, seed=1):
        # TODO(wujs): make it convinient
        # self.ACTION_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.ACTION_SPACE = gym.spaces.Discrete(5)
        self.OBSERVATION_SPACE = gym.spaces.Box(low=0, high=1, shape=(80, 80, 9))
        self.AGENT_ID = 'Agent-sheng'
        self.states = np.zeros(shape=(80, 80, 9))

        # TODO(wujs): make global var to specific scenario
        if 0 == len(scenario_path):
            # default path
            # self.scenario_path = ['scenarios/left_turn']
            scenario_path = ['scenarios/roundabout']
        
        sstudio.build_scenario(scenario_path)

        max_episode_steps = 600

        # define agent interface
        agent_interface = AgentInterface(
            max_episode_steps=max_episode_steps,
            waypoints=True,
            neighborhood_vehicles=NeighborhoodVehicles(radius=60),
            rgb=RGB(80, 80, 32/80), 
            action=ActionSpaceType.LaneWithContinuousSpeed,
            event_configuration=EventConfiguration(not_moving_time=20, not_moving_distance=1),
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
                                headless= not envision,
                                sumo_headless= not sumo,
                                visdom=visdom,
                                seed=seed, 
                                )
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
        speed_reward = env_obs.ego_vehicle_state.speed * 0.001

        reward = 0
        # if env_obs.events.reached_goal:
        #     reward += 10

        # if env_obs.events.collisions:
        #     reward -= 30

        # if env_obs.events.off_road:
        #     reward -= 5

        # if env_obs.events.on_shoulder:
        #     reward -= 2

        if env_obs.events.not_moving:
            reward -= 1

        if env_obs.events.reached_goal:
            reward += 1
        
        if env_obs.events.collisions:
            reward -= 1

        return reward + speed_reward

    # action space
    def action_adapter(self, model_action):
        speed = 10
        lane = 0
        
        if model_action == 0:
            pass
        elif model_action == 1:
            lane -= 1
        elif model_action == 2:
            lane += 1
        elif model_action == 3:
            speed = 0
        elif model_action == 4:
            speed = 4
        
        # print(speed, lane)
        
        return (speed, lane)

    # def action_adapter(self, model_action):
    #     action_map = ["keep_lane", "slow_down", "change_lane_left", "change_lane_right"]
    #     return action_map[model_action]

    # information
    def info_adapter(self, observation, reward, info):
        return info