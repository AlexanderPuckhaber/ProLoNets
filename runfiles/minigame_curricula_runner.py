import sc2
from sc2 import Race, Difficulty
import os
import sys

from sc2.constants import *
from sc2.position import Pointlike, Point2
from sc2.player import Bot, Computer
from sc2.unit import Unit as sc2Unit
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, '../../')
import torch
from agents.prolonet_agent import DeepProLoNet
from agents.heuristic_agent import RandomHeuristic, StarCraftMicroHeuristic
from agents.py_djinn_agent import DJINNAgent
from agents.lstm_agent import LSTMNet
from agents.baseline_agent import FCNet
from opt_helpers.replay_buffer import discount_reward
from runfiles import sc_helpers
import numpy as np
import torch.multiprocessing as mp
import argparse
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

DEBUG = False
SUPER_DEBUG = False
if SUPER_DEBUG:
    DEBUG = True

FAILED_REWARD = -0.0
SUCCESS_BUILD_REWARD = 1.
SUCCESS_TRAIN_REWARD = 1.
SUCCESS_SCOUT_REWARD = 1.
SUCCESS_ATTACK_REWARD = 1.
SUCCESS_MINING_REWARD = 1.

maps = ['DefeatRoaches', 'DefeatRoaches']
temperature = 1.0
learning_rate = 0.01
window_size = 5
n_tasks = len(maps)
np.random.seed(42)

Q_estimates = np.zeros(len(maps))


def update_weight(idx, total_return):
    learning_rate * total_return + (1 - learning_rate) * Q_estimates[idx]


def get_probability_task(idx):
    np.exp(Q_estimates[idx] / temperature) / np.sum(np.exp(Q_estimates / temperature))


def get_task():
    p = [get_probability_task(idx) for idx in range(n_tasks)]
    return np.random.choice(list(range(n_tasks)), p=p)


def update_weight_at_idx(queue, idx):
    regression = LinearRegression()
    x = np.array(list(range(window_size))).reshape(1, -1)
    regression.fit(x, queue[idx])
    y_hat = regression.predict(list(range(window_size)))
    reg_coef = r2_score(queue[idx], y_hat)
    update_weight(idx, reg_coef)

#
# def window_algorithm():
#     queue = [[] for _ in range(n_tasks)]
#     total_steps = 1000
#
#     for idx in range(n_tasks):
#         for _ in range(window_size):
#             # perform task at idx here
#             current_return = None
#             queue[idx].insert(0, current_return)
#         update_weight_at_idx(queue, idx)
#
#     for i in range(total_steps):
#         next_task_idx = get_task()
#         # perform task here
#         current_return = None
#         queue[next_task_idx].pop()
#         queue[next_task_idx].insert(0, current_return)
#         update_weight_at_idx(queue, next_task_idx)


class SC2MicroBot(sc2.BotAI):
    def __init__(self, rl_agent, kill_reward=1):
        super(SC2MicroBot, self).__init__()
        self.agent = rl_agent
        self.kill_reward = kill_reward
        self.action_buffer = []
        self.prev_state = None
        self.last_known_enemy_units = []
        self.itercount = 0
        self.last_reward = 0
        self.my_tags = None
        self.agent_list = []
        self.dead_agent_list = []
        self.dead_index_mover = 0
        self.dead_enemies = 0

    async def on_step(self, iteration):

        if iteration == 0:
            self.my_tags = [unit.tag for unit in self.units]
            for unit in self.units:
                self.agent_list.append(self.agent.duplicate())
        else:
            self.last_reward = 0
            for unit in self.state.dead_units:
                if unit in self.my_tags:
                    self.last_reward -= 1
                    self.dead_agent_list.append(self.agent_list[self.my_tags.index(unit)])
                    del self.agent_list[self.my_tags.index(unit)]
                    del self.my_tags[self.my_tags.index(unit)]
                    self.dead_agent_list[-1].save_reward(self.last_reward)
                else:
                    self.last_reward += self.kill_reward
                    self.dead_enemies += 1
            # if len(self.state.dead_units) > 0:
            for agent in self.agent_list:
                agent.save_reward(self.last_reward)
        for unit in self.units:
            if unit.tag not in self.my_tags:
                self.my_tags.append(unit.tag)
                self.agent_list.append(self.agent.duplicate())
        # if iteration % 20 != 0:
        #     return
        all_unit_data = []
        for unit in self.units:
            all_unit_data.append(sc_helpers.get_unit_data(unit))
        while len(all_unit_data) < 3:
            all_unit_data.append([-1, -1, -1, -1])
        for unit, agent in zip(self.units, self.agent_list):
            nearest_allies = sc_helpers.get_nearest_enemies(unit, self.units)
            unit_data = sc_helpers.get_unit_data(unit)
            nearest_enemies = sc_helpers.get_nearest_enemies(unit, self.known_enemy_units)
            unit_data = np.array(unit_data).reshape(-1)
            enemy_data = []
            allied_data = []
            for enemy in nearest_enemies:
                enemy_data.extend(sc_helpers.get_enemy_unit_data(enemy))
            for ally in nearest_allies[1:3]:
                allied_data.extend(sc_helpers.get_unit_data(ally))
            enemy_data = np.array(enemy_data).reshape(-1)
            allied_data = np.array(allied_data).reshape(-1)
            state_in = np.concatenate((unit_data, allied_data, enemy_data))
            action = agent.get_action(state_in)
            await self.execute_unit_action(unit, action, nearest_enemies)
        try:
            await self.do_actions(self.action_buffer)
        except sc2.protocol.ProtocolError:
            print("Not in game?")
            self.action_buffer = []
            return
        self.action_buffer = []

    async def execute_unit_action(self, unit_in, action_in, nearest_enemies):
        if action_in < 4:
            await self.move_unit(unit_in, action_in)
        elif action_in < 9:
            await self.attack_nearest(unit_in, action_in, nearest_enemies)
        else:
            pass

    async def move_unit(self, unit_to_move, direction):
        current_pos = unit_to_move.position
        target_destination = current_pos
        if direction == 0:
            target_destination = [current_pos.x, current_pos.y + 5]
        elif direction == 1:
            target_destination = [current_pos.x + 5, current_pos.y]
        elif direction == 2:
            target_destination = [current_pos.x, current_pos.y - 5]
        elif direction == 3:
            target_destination = [current_pos.x - 5, current_pos.y]
        self.action_buffer.append(unit_to_move.move(Point2(Pointlike(target_destination))))

    async def attack_nearest(self, unit_to_attack, action_in, nearest_enemies_list):
        if len(nearest_enemies_list) > action_in-4:
            target = nearest_enemies_list[action_in-4]
            if target is None:
                return -1
            self.action_buffer.append(unit_to_attack.attack(target))
        else:
            return -1

    def finish_episode(self, game_result):
        print("Game over!")
        if game_result == sc2.Result.Defeat:
            for index in range(len(self.agent_list), 0, -1):
                self.dead_agent_list.append(self.agent_list[index-1])
                self.dead_agent_list[-1].save_reward(-1)
            del self.agent_list[:]
        elif game_result == sc2.Result.Tie:
            reward = 0
        elif game_result == sc2.Result.Victory:
            reward = 0  # - min(self.itercount/500.0, 900) + self.units.amount
        else:
            # ???
            return -13
        if len(self.agent_list) > 0:
            reward_sum = sum(self.agent_list[0].replay_buffer.rewards_list)
        else:
            reward_sum = sum(self.dead_agent_list[-1].replay_buffer.rewards_list)

        for agent_index in range(len(self.agent_list)):
            rewards_list, advantage_list, deeper_advantage_list = discount_reward(
                self.agent_list[agent_index].replay_buffer.rewards_list,
                self.agent_list[agent_index].replay_buffer.value_list,
                self.agent_list[agent_index].replay_buffer.deeper_value_list)
            self.agent_list[agent_index].replay_buffer.rewards_list = rewards_list
            self.agent_list[agent_index].replay_buffer.advantage_list = advantage_list
            self.agent_list[agent_index].replay_buffer.deeper_advantage_list = deeper_advantage_list
        for dead_agent_index in range(len(self.dead_agent_list)):
            rewards_list, advantage_list, deeper_advantage_list = discount_reward(
                self.dead_agent_list[dead_agent_index].replay_buffer.rewards_list,
                self.dead_agent_list[dead_agent_index].replay_buffer.value_list,
                self.dead_agent_list[dead_agent_index].replay_buffer.deeper_value_list)
            self.dead_agent_list[dead_agent_index].replay_buffer.rewards_list = rewards_list
            self.dead_agent_list[dead_agent_index].replay_buffer.advantage_list = advantage_list
            self.dead_agent_list[dead_agent_index].replay_buffer.deeper_advantage_list = deeper_advantage_list
        return self.dead_enemies*self.kill_reward - len(self.dead_agent_list)


def run_episode(q, main_agent, game_mode):
    result = None
    agent_in = main_agent
    kill_reward = 1
    if 'DefeatRoaches' in game_mode:
        kill_reward = 10
    elif 'DefeatZerglingsAndBanelings' in game_mode:
        kill_reward = 5
    bot = SC2MicroBot(rl_agent=agent_in, kill_reward=kill_reward)

    try:
        result = sc2.run_game(sc2.maps.get(game_mode),
                              [Bot(Race.Terran, bot)],
                              realtime=False)
    except KeyboardInterrupt:
        result = [-1, -1]
    except Exception as e:
        print(str(e))
        print("No worries", e, " carry on please")
    if type(result) == list and len(result) > 1:
        result = result[0]
    reward_sum = bot.finish_episode(result)
    for agent in bot.agent_list+bot.dead_agent_list:
        agent_in.replay_buffer.extend(agent.replay_buffer.__getstate__())
    if q is not None:
        try:
            q.put([reward_sum, agent_in.replay_buffer.__getstate__()])
        except RuntimeError as e:
            print(e)
            return [reward_sum, agent_in.replay_buffer.__getstate__()]
    return [reward_sum, agent_in.replay_buffer.__getstate__()]

def main(episodes, agent, num_processes, game_mode):
    running_reward_array = []
    # lowered = False

    queue = [[] for _ in range(n_tasks)]
    successful_runs = 0

    for idx in range(n_tasks):
        print('Gathering initial window for task', idx)
        for _ in range(window_size):
            master_reward, reward, running_reward = 0, 0, 0
            try:
                returned_object = run_episode(None, main_agent=agent, game_mode=maps[idx])
                master_reward += returned_object[0]
                current_return = returned_object[0]
                running_reward_array.append(returned_object[0])
                # agent.replay_buffer.extend(returned_object[1])
                successful_runs += 1
            except MemoryError as e:
                print(e)
                continue
            reward = master_reward / float(successful_runs)
            agent.end_episode(reward, num_processes)

            queue[idx].insert(0, current_return)
        update_weight_at_idx(queue, idx)
        print('Finished initial window for task', idx)


    for episode in range(1, episodes+1):
        successful_runs = 0
        master_reward, reward, running_reward = 0, 0, 0
        processes = []
        try:
            next_task_idx = get_task()
            print('Running task', next_task_idx)
            returned_object = run_episode(None, main_agent=agent, game_mode=maps[next_task_idx])
            master_reward += returned_object[0]
            current_return = returned_object[0]
            queue[next_task_idx].pop()
            queue[next_task_idx].insert(0, current_return)
            update_weight_at_idx(queue, next_task_idx)
            running_reward_array.append(returned_object[0])
            # agent.replay_buffer.extend(returned_object[1])
            successful_runs += 1
        except MemoryError as e:
            print('Memory error.')
            print(e)
            continue
        reward = master_reward / float(successful_runs)
        agent.end_episode(reward, num_processes)
        running_reward = sum(running_reward_array[-100:]) / float(min(100.0, len(running_reward_array)))
        if episode % 50 == 0:
            print(f'Episode {episode}  Last Reward: {reward}  Average Reward: {running_reward}')
            print(f"Running {num_processes} concurrent simulations per episode")
        if episode % 300 == 0:
            agent.save('../models/' + str(episode) + 'th')
            agent.lower_lr()

    return running_reward_array


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent_type", help="architecture of agent to run", type=str, default='prolo')
    parser.add_argument("-env", "--env_type", help="FindAndDefeatZerglings, DefeatRoaches, DefeatZerglingsAndBanelings",
                        type=str, default='FindAndDefeatZerglings')
    parser.add_argument("-e", "--episodes", help="how many episodes", type=int, default=1000)
    parser.add_argument("-p", "--processes", help="how many processes?", type=int, default=1)
    parser.add_argument("-gpu", help="run on GPU?", action='store_true')
    parser.add_argument("-vec", help="Vectorized ProLoNet?", action='store_true')
    parser.add_argument("-adv", help="Adversarial ProLoNet?", action='store_true')
    parser.add_argument("-rand", help="Random ProLoNet?", action='store_true')
    parser.add_argument("-deep", help="Deepening?", action='store_true')
    parser.add_argument("-s", "--sl_init", help="sl to rl for fc net?", action='store_true')

    args = parser.parse_args()
    AGENT_TYPE = args.agent_type  # 'shallow_prolo', 'prolo', 'random', 'fc', 'lstm'
    ADVERSARIAL = args.adv  # Adversarial prolo, applies for AGENT_TYPE=='shallow_prolo'
    SL_INIT = args.sl_init  # SL->RL fc, applies only for AGENT_TYPE=='fc'
    NUM_EPS = args.episodes  # num episodes Default 1000
    NUM_PROCS = args.processes  # num concurrent processes Default 1
    USE_GPU = args.gpu  # Applies for 'prolo' only. use gpu? Default false
    VECTORIZED = args.vec  # Applies for 'prolo' vectorized or no? Default false
    RANDOM = args.rand  # Applies for 'prolo' random init or no? Default false
    DEEPEN = args.deep  # Applies for 'prolo' deepen or no? Default false
    ENV_TYPE = args.env_type
    torch.set_num_threads(NUM_PROCS)
    dim_in = 37
    dim_out = 10
    bot_name = AGENT_TYPE + ENV_TYPE
    # mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')
    for _ in range(5):

        if AGENT_TYPE == 'prolo':
            policy_agent = DeepProLoNet(distribution='one_hot',
                                        bot_name=bot_name,
                                        input_dim=dim_in,
                                        output_dim=dim_out,
                                        use_gpu=USE_GPU,
                                        vectorized=VECTORIZED,
                                        randomized=RANDOM,
                                        adversarial=ADVERSARIAL,
                                        deepen=DEEPEN)

        elif AGENT_TYPE == 'fc':
            policy_agent = FCNet(input_dim=dim_in,
                                 bot_name=bot_name,
                                 output_dim=dim_out,
                                 sl_init=SL_INIT,
                                 num_hidden=1)
        elif AGENT_TYPE == 'lstm':
            policy_agent = LSTMNet(input_dim=dim_in,
                                   bot_name=bot_name,
                                   output_dim=dim_out)
        elif AGENT_TYPE == 'random':
            policy_agent = RandomHeuristic(bot_name=bot_name,
                                           action_dim=dim_out)
        elif AGENT_TYPE == 'heuristic':
            policy_agent = StarCraftMicroHeuristic(bot_name=bot_name)
        elif AGENT_TYPE == 'djinn':
            policy_agent = DJINNAgent(bot_name=bot_name,
                                      input_dim=dim_in,
                                      output_dim=dim_out)
        else:
            raise Exception('No valid network selected')
        main(episodes=NUM_EPS, agent=policy_agent, num_processes=NUM_PROCS, game_mode=ENV_TYPE)
