
import secrets
import gym
import numpy as np
import glob
import random
import time

from typing import Dict
from smarts.env.hiway_env import HiWayEnv
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import NeighborhoodVehicles, RGB
from smarts.core.controllers import ActionSpaceType
from smarts.core.sensors import Observation
from smarts.core.utils.math import vec_2d


class SmartsEnv():
    def __init__(self, scenario_path=[]):
        # TODO(wujs): make it convinient
        # self.ACTION_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.ACTION_SPACE = gym.spaces.Discrete(5)
        self.OBSERVATION_SPACE = gym.spaces.Box(
            low=0, high=1, shape=(80, 80, 9))
        self.AGENT_ID = 'Agent-sheng'
        self.states = np.zeros(shape=(80, 80, 9))

        # TODO(wujs): make global var to specific scenario
        # scenario_path = ['scenarios/left_turn']
        scenario_path = ['scenarios/roundabout']
        max_episode_steps = 600

        # define agent interface
        agent_interface = AgentInterface(
            max_episode_steps=max_episode_steps,
            waypoints=True,
            neighborhood_vehicles=NeighborhoodVehicles(radius=60),
            rgb=RGB(80, 80, 32/80),
            action=ActionSpaceType.LaneWithContinuousSpeed,
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
                             visdom=True,
                             sumo_headless=False)
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
        def lane_ttc(obs: Observation) -> Dict[str, np.ndarray]:
            ego = obs.ego_vehicle_state
            waypoint_paths = obs.waypoint_paths
            wps = [path[0] for path in waypoint_paths]

            # distance of vehicle from center of lane
            closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
            signed_dist_from_center = closest_wp.signed_lateral_error(
                ego.position)
            lane_hwidth = closest_wp.lane_width * 0.5
            norm_dist_from_center = signed_dist_from_center / lane_hwidth

            ego_ttc, ego_lane_dist = _ego_ttc_lane_dist(
                obs, closest_wp.lane_index)

            return {
                "distance_from_center": np.array([norm_dist_from_center]),
                "angle_error": np.array([closest_wp.relative_heading(ego.heading)]),
                "speed": np.array([ego.speed]),
                "steering": np.array([ego.steering]),
                "ego_ttc": np.array(ego_ttc),
                "ego_lane_dist": np.array(ego_lane_dist),
            }

        def _ego_ttc_lane_dist(obs: Observation, ego_lane_index: int):
            ttc_by_p, lane_dist_by_p = _ttc_by_path(obs)
            return _ego_ttc_calc(ego_lane_index, ttc_by_p, lane_dist_by_p)

        def _ttc_by_path(obs: Observation):
            ego = obs.ego_vehicle_state
            waypoint_paths = obs.waypoint_paths
            neighborhood_vehicle_states = obs.neighborhood_vehicle_states

            # first sum up the distance between waypoints along a path
            # ie. [(wp1, path1, 0),
            #      (wp2, path1, 0 + dist(wp1, wp2)),
            #      (wp3, path1, 0 + dist(wp1, wp2) + dist(wp2, wp3))]

            wps_with_lane_dist = []
            for path_idx, path in enumerate(waypoint_paths):
                lane_dist = 0.0
                for w1, w2 in zip(path, path[1:]):
                    wps_with_lane_dist.append((w1, path_idx, lane_dist))
                    lane_dist += np.linalg.norm(w2.pos - w1.pos)
                wps_with_lane_dist.append((path[-1], path_idx, lane_dist))

            # next we compute the TTC along each of the paths
            ttc_by_path_index = [1000] * len(waypoint_paths)
            lane_dist_by_path_index = [1] * len(waypoint_paths)

            for v in neighborhood_vehicle_states:
                # find all waypoints that are on the same lane as this vehicle
                wps_on_lane = [
                    (wp, path_idx, dist)
                    for wp, path_idx, dist in wps_with_lane_dist
                    if wp.lane_id == v.lane_id
                ]

                if not wps_on_lane:
                    # this vehicle is not on a nearby lane
                    continue

                # find the closest waypoint on this lane to this vehicle
                nearest_wp, path_idx, lane_dist = min(
                    wps_on_lane, key=lambda tup: np.linalg.norm(
                        tup[0].pos - vec_2d(v.position))
                )

                if np.linalg.norm(nearest_wp.pos - vec_2d(v.position)) > 2:
                    # this vehicle is not close enough to the path, this can happen
                    # if the vehicle is behind the ego, or ahead past the end of
                    # the waypoints
                    continue

                relative_speed_m_per_s = (ego.speed - v.speed) * 1000 / 3600
                if abs(relative_speed_m_per_s) < 1e-5:
                    relative_speed_m_per_s = 1e-5

                ttc = lane_dist / relative_speed_m_per_s
                ttc /= 10
                if ttc <= 0:
                    # discard collisions that would have happened in the past
                    continue

                lane_dist /= 100
                lane_dist_by_path_index[path_idx] = min(
                    lane_dist_by_path_index[path_idx], lane_dist
                )
                ttc_by_path_index[path_idx] = min(
                    ttc_by_path_index[path_idx], ttc)

            return ttc_by_path_index, lane_dist_by_path_index

        def _ego_ttc_calc(ego_lane_index: int, ttc_by_path, lane_dist_by_path):
            ego_ttc = [0] * 3
            ego_lane_dist = [0] * 3

            ego_ttc[1] = ttc_by_path[ego_lane_index]
            ego_lane_dist[1] = lane_dist_by_path[ego_lane_index]

            max_lane_index = len(ttc_by_path) - 1
            min_lane_index = 0
            if ego_lane_index + 1 > max_lane_index:
                ego_ttc[2] = 0
                ego_lane_dist[2] = 0
            else:
                ego_ttc[2] = ttc_by_path[ego_lane_index + 1]
                ego_lane_dist[2] = lane_dist_by_path[ego_lane_index + 1]
            if ego_lane_index - 1 < min_lane_index:
                ego_ttc[0] = 0
                ego_lane_dist[0] = 0
            else:
                ego_ttc[0] = ttc_by_path[ego_lane_index - 1]
                ego_lane_dist[0] = lane_dist_by_path[ego_lane_index - 1]
            return ego_ttc, ego_lane_dist

        # *================================== Calculate ==================================
        dic = lane_ttc(env_obs)
        # ego_ttc
        ego_ttc = 0
        for i in dic['ego_ttc']:
            if (1000-i) > 0.001:
                ego_ttc += i
            else:
                ego_ttc += 5

        # ego_lane_dist
        ego_lane_dist = 0
        for i in dic['ego_lane_dist']:
            ego_lane_dist += i

        progress = env_obs.ego_vehicle_state.speed
        goal = 150 if env_obs.events.reached_goal else - 1
        crash = -20 if env_obs.events.collisions else 0

        reward = 0.01 * progress + goal + crash + \
            env_reward + 0.05 * ego_ttc + 0.05 * ego_lane_dist

        if env_obs.events.off_road:
            print("Event: Off Road!")
            reward -= 5

        if env_obs.events.on_shoulder:
            print("Event: On Shoulder!")
            reward -= 5

        if env_obs.events.reached_goal:
            print("Event: Reach the Goal!")

        if env_obs.events.collisions:
            print("Event: Crash!")

        return reward

    # def action_adapter(self, model_action):
    #     action_map = ["keep_lane", "slow_down",
    #                   "change_lane_left", "change_lane_right"]
    #     return action_map[model_action]
    def action_adapter(self, model_action):
        speed = 8
        lane = 0

        if model_action == 1:
            lane -= 1
            print("Action: Change lane!")
        elif model_action == 2:
            lane += 1
            print("Action: Change lane!")

        if model_action == 3:
            speed = 0
            print("Action: Stop!")
        elif model_action == 4:
            speed = 12
            print("Action: Speed up!")

        return (speed, lane)

    # information

    def info_adapter(self, observation, reward, info):
        return info
