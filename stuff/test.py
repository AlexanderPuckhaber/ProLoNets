
import os
import sys

import torchviz
import numpy as np

sys.path.insert(0, os.path.abspath('../'))
import torch
from agents.prolonet_agent import DeepProLoNet
from agents.py_djinn_agent import DJINNAgent
from agents.lstm_agent import LSTMNet
from agents.baseline_agent import FCNet

import time
import torch.multiprocessing as mp
import torch.nn as nn
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent_type", help="architecture of agent to run", type=str, default='djinn')
    parser.add_argument("-e", "--episodes", help="how many episodes", type=int, default=1000)
    parser.add_argument("-p", "--processes", help="how many processes?", type=int, default=1)
    parser.add_argument("-env", "--env_type", help="environment to run on", type=str, default='cart')
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
    ENV_TYPE = args.env_type  # 'cart' or 'lunar' Default 'cart'
    USE_GPU = args.gpu  # Applies for 'prolo' only. use gpu? Default false
    VECTORIZED = args.vec  # Applies for 'prolo' vectorized or no? Default false
    RANDOM = args.rand  # Applies for 'prolo' random init or no? Default false
    DEEPEN = args.deep  # Applies for 'prolo' deepen or no? Default false
    # torch.set_num_threads(NUM_PROCS)
    dim_in = 194
    dim_out = 44
    bot_name = AGENT_TYPE + 'SC_Macro'+'Medium'
    mp.set_sharing_strategy('file_system')
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
                             sl_init=SL_INIT)
    elif AGENT_TYPE == 'lstm':
        policy_agent = LSTMNet(input_dim=dim_in,
                               bot_name=bot_name,
                               output_dim=dim_out)
    elif AGENT_TYPE == 'djinn':
        policy_agent = DJINNAgent(bot_name=bot_name,
                                  input_dim=dim_in,
                                  output_dim=dim_out)
    else:
        raise Exception('No valid network selected')
    start_time = time.time()

    # main(episodes=NUM_EPS, agent_in=policy_agent, num_processes=NUM_PROCS, reset_on_fail=True)
    policy_agent.load('../models/1')

    print(policy_agent.action_network.added_levels.named_parameters())

    model = policy_agent.value_network

    x = torch.randn(1, 194)
    y = model(x)

    print(y)

    torchviz.make_dot(var=y.mean(), params=dict(model.named_parameters())).render(bot_name, format='png')