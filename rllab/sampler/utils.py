import numpy as np
from rllab.misc import tensor_utils
import time

from collections import defaultdict

def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()

    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated:
        env.render(close=True)
        #env.render()

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )


def rollout_w_truth(env, agent, max_path_length=np.inf, animated=False,
        save_gif = False, speedup=1, mean=np.zeros(2), std=np.ones(2),
        seed = -1,
        **kwargs):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset(seed=seed)

    truth = defaultdict(list)
    ef = env.wrapped_env.j.rollout_ego_features(env.wrapped_env.simparams)
    for d in ef:
        for key, val in d.items():
            truth[key].append(val)

    agent.reset()
    path_length = 0
    if animated:
        env.render()
    if save_gif:
        initial_simparams0 = env.wrapped_env.copy_simparams()
        initial_simparams1 = env.wrapped_env.copy_simparams()

    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        a = (a * std) + mean

        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if save_gif:
        actions = [np.clip(action,
            *env.wrapped_env.j.action_space_bounds(initial_simparams0)) for action in actions]
        env.wrapped_env.save_gif(initial_simparams0, np.column_stack(actions),
                kwargs['filename'], truth_simparams = initial_simparams1)

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    ), truth


def gif_rollout(env, agent, max_path_length=np.inf, save_gif=False, **kwargs):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    initial_simparams = env.initial_simparams
    #agent.reset()
    path_length = 0
    #if animated:
    #    env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o.reshape(-1))
        rewards.append(r)
        actions.append(a.reshape(-1))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    if save_gif:
        actions = [np.clip(action,
            *env.j.action_space_bounds(initial_simparams)) for action in actions]
        env.save_gif(initial_simparams, np.column_stack(actions), kwargs['filename'])
        #env.render(close=True)
        #env.render()

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )
