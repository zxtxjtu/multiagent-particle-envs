import numpy as np
from torch.utils.tensorboard import SummaryWriter

from multiagent.charge_core import World, Pile, EV, WaitArea
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set the minimum time interval(1->one hour,0.25->15 min)
        world.slot = 0.5
        # set the maximal deal with grid
        world.max_deal = 1000
        # set the T
        world.cycle = 96
        # set piles
        num_piles = 3
        pile_max_out = 150
        pile_min_in = 0.1 * pile_max_out
        world.piles = [Pile() for _ in range(num_piles)]
        for pile in world.piles:
            pile.max_out = pile_max_out
            pile.min_in = pile_min_in
        # set ES
        es_max_c = 5000
        es_min_c = 0.01 * es_max_c
        es_max_out = 1000
        es_min_in = es_max_out
        world.es.max_c = es_max_c
        world.es.min_c = es_min_c
        world.es.max_out = es_max_out
        world.es.min_in = es_min_in
        # set arriving rate of EVs
        # set power of PV and price of grid
        world.pv.installed_capacity = 1000
        low_price = 0.4185
        flat_price = 1.1729
        high_price = 1.4503
        peak_price = 2.2520
        # np.random.seed(world.cycle // 24)
        power_1 = [0] * 12 + [24, 72, 155, 272, 373, 472, 548, 618, 663, 702, 745, 740, 761, 760, 745, 725, 711,
                              670, 620, 541, 452, 335, 226, 159, 53, 14] + [0] * 10
        power_2 = [0] * 12 + [60, 104, 159, 224, 299, 371, 449, 499, 544, 585, 637, 629, 638, 614, 632, 625, 596, 560,
                              530, 500, 432, 369, 253, 150, 117, 71, 43, 9] + [0] * 8
        power_3 = [0] * 13 + [68, 156, 257, 343, 425, 487, 542, 592, 619, 640, 638, 630, 630, 540, 557, 461, 344, 315,
                              224, 154, 56, 28] + [0] * 12
        power_4 = [0] * 16 + [19, 48, 64, 118, 163, 203, 238, 275, 286, 297, 234, 272, 267, 219, 73, 46, 36, 22] + [
            0] * 14
        # pv_noise = np.random.uniform(0.6, 1.1)
        world.pv.power += power_1
        # noise_power = [p * pv_noise for p in power_1]
        world.pv.power += power_2

        one_day_price = [low_price] * int(8 / world.slot) + [high_price] * int(1 / world.slot) + [peak_price] * int(
            2 / world.slot) + [flat_price] * int(2 / world.slot) + [high_price] * int(2 / world.slot) + [
                            peak_price] * int(
            2 / world.slot) + [high_price] * int(5 / world.slot) + [low_price] * int(2 / world.slot)

        arr_lamda = [1] * 14 + [2, 2, 4, 4, 8, 8, 4, 4, 2, 2] * 3 + [1] * 4

        for _ in range(int(world.cycle * world.slot // 24)):
            # pv_power = np.random.uniform(0.5, 0.99, 9)
            # pv_power = np.round(np.clip(pv_power, 0.3, 0.99), 3).tolist()
            # pv_power.sort()
            # pv_power_up = pv_power.copy()
            # pv_power.reverse()
            # pv_power_down = pv_power[1:5]
            # pv_power = [0] * 6 + pv_power_up + pv_power_down + [0] * 5
            # world.pv.power += [pv * world.pv.installed_capacity for pv in pv_power]

            diff_factor = np.random.uniform(0.95, 1.05)

            world.price.buy_from_grid_p += [price * diff_factor for price in one_day_price]
            world.price.sell_to_grid_p += [price * 0.9 * diff_factor for price in one_day_price]

            # np.random.seed(0)
            arr_factor = np.random.uniform(0.9, 1.5)

            arr_lamda = [la * arr_factor for la in arr_lamda]
            world.arr_lamda += arr_lamda

        writer = SummaryWriter()
        for i in range(world.cycle):
            writer.add_scalars('timeseries_curve', {'pv': world.pv.power[i % len(world.pv.power)],
                                                    'buy_prices': world.price.buy_from_grid_p[i],
                                                    'sell_prices': world.price.sell_to_grid_p[i],
                                                    'arr_lamda': world.arr_lamda[i]}, i)
        writer.close()

        coe_ev_price = 2
        world.price.coe_sell_to_ev = coe_ev_price
        world.price.coe_buy_from_ev = 0.5 * coe_ev_price

        self.reset_world(world)
        return world

    def reset_world(self, world):
        # np.random.seed(4)
        world.cur_t = 0
        world.extra_deal = 0.0
        world.piles_power_sum = 0
        world.wait = WaitArea()

        world.es.state.cur_c = np.random.randint(world.es.min_c, world.es.max_c)
        world.es.action = 0.0
        world.es.real_action = 0.0

        world.working_num = 0

        for pile in world.piles:
            pile.state = EV()
            pile.connected = False
            pile.action = 0.0
            pile.real_action = 0.0

        world.update_wait(world.arr_lamda[world.cur_t])
        world.update_piles()

    # reward still need to modify
    def reward(self, world: World):
        reward = 0.0
        for pile in world.piles:
            if pile.connected:
                if pile.real_action > 0:
                    bill = pile.real_action * world.price.coe_sell_to_ev * world.slot
                else:
                    bill = pile.real_action * world.price.coe_buy_from_ev * world.slot
                reward += bill
                pile.state.payment += np.around(bill, 3)
                # EV到dep_t的末尾离开
                if pile.state.dep_t == world.cur_t:
                    # bill = world.price.coe_buy_from_ev * min(pile.state.cur_b - pile.state.tar_b, 0)
                    bill = max(2.5 * min(pile.state.cur_b - pile.state.tar_b, 0), -150)
                    reward += bill
                    pile.state.payment += np.around(bill, 3)
        if world.extra_deal > 0:
            reward += world.price.sell_to_grid_p[
                          world.cur_t % len(world.price.sell_to_grid_p)] * world.extra_deal * world.slot
        else:
            reward += world.price.buy_from_grid_p[
                          world.cur_t % len(world.price.buy_from_grid_p)] * world.extra_deal * world.slot
        return reward

    def observation(self, agent, world: World):
        # price need to normalize?
        sell_max = max(world.price.sell_to_grid_p)
        sell_min = min(world.price.sell_to_grid_p)
        sell_now = world.price.sell_to_grid_p[world.cur_t % len(world.price.sell_to_grid_p)]
        # sell_now = round((sell_now - sell_min) / (sell_max - sell_min), 3)
        buy_max = max(world.price.buy_from_grid_p)
        buy_min = min(world.price.buy_from_grid_p)
        buy_now = world.price.buy_from_grid_p[world.cur_t % len(world.price.buy_from_grid_p)]
        # buy_now = round((buy_now - buy_min) / (buy_max - buy_min), 3)
        prices = [sell_now, buy_now]
        world_state = [world.es.state.cur_c / world.es.max_c,
                       world.pv.power[world.cur_t % len(world.pv.power)] / world.pv.installed_capacity]
        if isinstance(agent, Pile):
            if agent.connected:
                state = [(agent.state.dep_t - world.cur_t) / (agent.state.dep_t - agent.state.arr_t),
                         (agent.state.tar_b - agent.state.cur_b) / (agent.state.tar_b - agent.state.ini_b)]
                return state + world_state + prices
            # return [0] * 2 + world_state + prices
            return [0] * 6
        return world_state + prices
