import gym
import numpy as np

from multiagent.charge_core import World


class MultiPileEnv(gym.Env):
    def __init__(self, world: World, reward_callback=None, observation_callback=None, reset_callback=None):
        self.world = world
        self.piles = world.piles
        self.es = world.es
        self.agents = self.piles + [self.es]
        self.n_agents = len(self.agents)
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.reset_callback = reset_callback
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            charge_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            self.action_space.append(charge_action_space)
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(gym.spaces.Box(low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32))

    def world_step(self, action_n):
        for i, agent in enumerate(self.agents):
            agent.action = action_n[i]
        self.world.step()

    def step(self, action_n):
        n = len(self.agents)
        obs_n = []
        info = {'ev_payment': [],
                'energy_get': [],
                'energy_miss': [],
                'ev_save_time': [],
                'ev_code': []}

        reward = self.reward_callback(self.world)

        for pile in self.world.piles:
            if pile.connected and (pile.state.dep_t == self.world.cur_t or pile.state.tar_b <= pile.state.cur_b):
                info['ev_payment'].append(pile.state.payment)
                info['energy_get'].append(pile.state.cur_b - pile.state.ini_b)
                info['energy_miss'].append(pile.state.tar_b - pile.state.cur_b)
                info['ev_save_time'].append(pile.state.dep_t - self.world.cur_t)
                info['ev_code'].append(pile.state.code)

                pile.connected = False
                pile.state = None
                # pile.real_action = 0

        for i, ev in enumerate(self.world.wait.wait_EVs):
            if self.world.wait.is_wait[i]:
                if ev.dep_t == self.world.cur_t:
                    reward += max(-100, 10 * (ev.arr_t - ev.dep_t) / (ev.cur_b / ev.tar_b))
                    self.world.wait.is_wait[i] = False

        self.world.cur_t += 1

        self.world.update_wait(self.world.arr_lamda[self.world.cur_t % len(self.world.arr_lamda)])
        self.world.update_piles()

        still_wait_ev = 0
        for i, ev in enumerate(self.world.wait.wait_EVs):
            if self.world.wait.is_wait[i]:
                still_wait_ev += 1
        #  obs need wait ev information?
        self.world.wait.still_wait = still_wait_ev

        for agent in self.agents:
            obs_n.append(self.observation_callback(agent, self.world))

        if self.world.cur_t != self.world.cycle:
            done_n = [False] * n
        else:
            done_n = [True] * n
            for i, ev in enumerate(self.world.wait.wait_EVs):
                if self.world.wait.is_wait[i]:
                    reward += max(-100, 10 * (ev.arr_t - self.world.cur_t) / (ev.cur_b / ev.tar_b))
            if still_wait_ev > 0:
                print("still wait: %d" % still_wait_ev)
        reward_n = [reward] * n

        return obs_n, reward_n, done_n, info

    def reset(self):
        self.reset_callback(self.world)
        obs_n = []
        self.agents = self.world.piles + [self.world.es]
        for agent in self.agents:
            obs_n.append(self.observation_callback(agent, self.world))
        return obs_n
