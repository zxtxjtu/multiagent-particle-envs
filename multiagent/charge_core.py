from typing import List

import numpy as np


class EV(object):
    def __init__(self):
        # arrival/departure timeslot
        self.arr_t = None
        self.dep_t = None
        # initial/target/current battery
        self.full_b = None
        self.ini_b = None
        self.tar_b = None
        self.cur_b = None
        # EV's sensitivity to cost
        self.sensitivity = 0.0
        self.payment = 0.0
        self.code = None
        self.pile_code = -1


class WaitArea(object):
    def __init__(self):
        self.wait_EVs: List[EV] = []
        self.is_wait = []
        self.max_accept = None
        self.still_wait = 0


class Pile(object):
    def __init__(self):
        # out means power limit transfer energy from pile to EV,which is positive
        self.max_out = None
        # in means power limit transfer energy from EV to pile,which is negative
        self.min_in = None
        # charge/discharge power
        self.action = None
        self.real_action = None
        self.connected = False
        self.state = EV()

    def modify_pile_action(self, slot=1.0):
        if self.action is None or self.connected is False:
            self.action = 0.0
            self.real_action = 0.0
        else:
            real_action = 0.0
            action = self.action[0]
            if self.action > 0:
                real_action = action * self.max_out
                real_action = min((self.state.tar_b - self.state.cur_b) / slot, real_action)
            else:
                real_action = action * self.min_in
                real_action = max((0 - self.state.cur_b) / slot, real_action)
            self.real_action = real_action

    def update_ev_state(self, slot=1.0):
        self.modify_pile_action(slot)
        if self.connected:
            self.state.cur_b += self.real_action * slot


# energy storge
class ESState(object):
    def __init__(self):
        # current capacity
        self.cur_c = None


class ES(object):
    def __init__(self):
        # max/min capacity
        self.max_c = None
        self.min_c = None
        # max/min transfer
        self.max_out = None
        self.min_in = None
        self.action = None
        self.real_action = None
        self.state = ESState()

    def modify_es_action(self, slot=1.0):
        if self.action is None:
            self.action = 0.0
            self.real_action = 0.0
        else:
            real_action = 0.0
            action = self.action[0]
            if self.action > 0:
                real_action = self.max_out * action
                real_action = min((self.state.cur_c - self.min_c) / slot, real_action)
            else:
                real_action = self.min_in * action
                real_action = max((self.state.cur_c - self.max_c) / slot, real_action)
            self.real_action = real_action

    def update_es_state(self, slot=1.0):
        self.modify_es_action(slot)
        self.state.cur_c -= self.real_action * slot


class PV(object):
    def __init__(self):
        self.installed_capacity = None
        self.power = []


class Price(object):
    def __init__(self):
        self.buy_from_grid_p = []
        self.sell_to_grid_p = []
        self.coe_sell_to_ev = None
        self.coe_buy_from_ev = None


class World(object):
    def __init__(self):
        # current timeslot of charge station
        self.cur_t = 0
        self.pv = PV()
        self.price = Price()
        self.piles: List[Pile] = []
        self.working_num = 0
        self.piles_power_sum = 0
        self.es = ES()
        self.extra_deal = 0.0
        self.max_deal = None
        self.wait = WaitArea()
        self.arr_lamda = []
        self.slot = 1.0
        self.cycle = None

    def step(self):
        curr_time = self.cur_t
        self.piles_power_sum = 0
        self.extra_deal = 0
        for pile in self.piles:
            if pile.connected:
                pile.update_ev_state(self.slot)
                self.piles_power_sum += pile.real_action
        self.es.update_es_state(self.slot)
        self.extra_deal = self.pv.power[curr_time % len(self.pv.power)] + self.es.real_action - self.piles_power_sum

    def update_piles(self):
        for pile_code, pile in enumerate(self.piles):
            if pile.connected:
                continue
            for i, ev in enumerate(self.wait.wait_EVs):
                if self.wait.is_wait[i] is False:
                    continue
                else:
                    pile.connected = True
                    pile.state = ev
                    self.wait.is_wait[i] = False
                    ev.pile_code = pile_code
                    break
        self.working_num = 0
        for pile in self.piles:
            if pile.connected:
                self.working_num += 1
            # if not pile.connected and len(self.wait.wait_EVs) > 0:
            #     if self.wait.is_wait[0] is False:
            #         self.wait.is_wait.pop(0)
            #         self.wait.wait_EVs.pop(0)
            #     else:
            #         pile.connected = self.wait.is_wait.pop(0)
            #         pile.state = self.wait.wait_EVs.pop(0)
            #
            # if len(self.wait.wait_EVs) == 0:
            #     break

    def update_wait(self, lamda):
        arr_num = np.random.poisson(lam=lamda, size=1)
        arr_num = int(arr_num)
        # np.random.seed(arr_num)
        for _ in range(arr_num):
            new_ev = EV()
            new_ev.arr_t = self.cur_t
            new_ev.dep_t = new_ev.arr_t + np.random.randint(1, int(3 // self.slot))
            new_ev.full_b = 100
            new_ev.ini_b = round(np.random.uniform(0.05, 0.25), 2) * new_ev.full_b
            new_ev.cur_b = new_ev.ini_b
            new_ev.tar_b = round(np.random.uniform(0.80, 0.99), 2) * new_ev.full_b
            # 可以为每一辆电车的焦虑建模
            new_ev.sensitivity = np.random.randint(0, 5)
            new_ev.payment = 0
            new_ev.code = len(self.wait.wait_EVs) + 1
            self.wait.wait_EVs.append(new_ev)
            self.wait.is_wait.append(True)
