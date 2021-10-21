import json
import logging
import math
import os
import time

import gym
import numpy as np
import torch as th
from abc import abstractmethod
from datetime import datetime
from statistics import mean

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn.modules.activation import Tanh, ReLU, ELU
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.ppo import CnnPolicy

verbose = True
REALLY_BIG_INT = 1_000_000_000

if verbose:
    logging.basicConfig(level=logging.DEBUG)


class TrainingSession:
    def __init__(self, params, path, seed):
        # initialize logger
        self.logger = logging.getLogger('{0}.{1}'.format(__name__,
                                                         type(self).__name__))

        # initialize results
        self.checkpoints = []
        self.win_rates = []
        self.episode_lengths = []
        self.action_histograms = []
        self.start_time, self.end_time = None, None

        # save parameters
        self.params = params
        self.path = os.path.dirname(__file__) + "/../../" + path
        self.seed = seed

    @abstractmethod
    def _train(self):
        pass

    def _save_results(self):
        results_path = self.path + '/results.txt'

        with open(results_path, 'w') as file:
            info = dict(**self.params, seed=self.seed, checkpoints=self.checkpoints,
                        win_rates=self.win_rates, ep_lengths=self.episode_lengths,
                        action_histograms=self.action_histograms,
                        start_time=str(self.start_time), end_time=str(self.end_time))
            info = json.dumps(info, indent=2)

            file.write(info)

        self.logger.debug(f"Results saved at {results_path}.")

    def run(self):
        # log start time
        self.start_time = datetime.now()
        self.logger.info("Training...")

        # do the training
        self._train()

        # log end time
        self.end_time = datetime.now()
        self.logger.info(f"End of training. Time elapsed: {self.end_time - self.start_time}.")

        # save model info to results file
        self._save_results()


class FixedAdversary(TrainingSession):
    def __init__(self, model_builder, model_params, env_builder, env_params, eval_env_params,
                 train_episodes, eval_episodes, num_evals, play_first, path,
                 seed, num_envs=1):
        super(FixedAdversary, self).__init__(model_params, path, seed)

        # log start time
        start_time = time.perf_counter()

        # initialize parallel environments
        self.logger.debug("Initializing training env...")
        env = []

        for i in range(num_envs):
            # no overlap between episodes at each concurrent env
            if seed is not None:
                current_seed = seed + (train_episodes // num_envs) * i
            else:
                current_seed = None

            # create the env
            env.append(lambda: env_builder(**env_params))

        # wrap envs in a vectorized env
        self.env: VecEnv = DummyVecEnv(env)

        # initialize evaluator
        self.logger.debug("Initializing evaluator...")
        eval_seed = seed + train_episodes if seed is not None else None
        self.evaluator: Evaluator = Evaluator(env_builder, eval_env_params,
                                              eval_episodes, eval_seed, num_envs)

        # build the model
        self.logger.debug("Building the model...")
        self.model = model_builder(self.env, seed, **model_params)

        # create necessary folders
        os.makedirs(self.path, exist_ok=True)

        # set tensorflow log dir
        self.model.tensorflow_log = self.path

        # save parameters
        self.train_episodes = train_episodes
        self.num_evals = num_evals
        self.eval_frequency = train_episodes / num_evals

        # initialize control attributes
        self.model.last_eval = None
        self.model.next_eval = 0
        self.model.role_id = 0 if play_first else 1

        # log end time
        end_time = time.perf_counter()

        self.logger.debug("Finished initializing training session "
                          f"({round(end_time - start_time, ndigits=3)}s).")

    def _training_callback(self, _locals=None, _globals=None):
        episodes_so_far = sum(self.env.get_attr('episodes'))

        # if it is time to evaluate, do so
        if episodes_so_far >= self.model.next_eval:
            # save model
            model_path = self.path + f'/{episodes_so_far}'
            self.model.save(model_path)
            save_model_as_json(self.model, self.params['activation'], model_path)
            self.logger.debug(f"Saved model at {model_path}.zip/json.")

            # evaluate the model
            self.logger.info(f"Evaluating model ({episodes_so_far} episodes)...")
            start_time = time.perf_counter()

            mean_reward, ep_length, act_hist = \
                self.evaluator.run(self.model, play_first=self.model.role_id == 0)

            end_time = time.perf_counter()
            self.logger.info(f"Finished evaluating "
                             f"({round(end_time - start_time, 3)}s). "
                             f"Avg. reward: {mean_reward}")

            # save the results
            self.checkpoints.append(episodes_so_far)
            self.win_rates.append((mean_reward + 1) / 2)
            self.episode_lengths.append(ep_length)
            self.action_histograms.append(act_hist)

            # update control attributes
            self.model.last_eval = episodes_so_far
            self.model.next_eval += self.eval_frequency

            # write partial results to file
            self._save_results()

        # if training should end, return False to end training
        training_is_finished = episodes_so_far >= self.train_episodes

        if training_is_finished:
            self.logger.debug(f"Training ended at {episodes_so_far} episodes")

        return not training_is_finished

    def _train(self):
        # save and evaluate starting model
        self._training_callback()

        try:
            # train the model
            # note: dynamic learning or clip rates will require accurate # of timesteps
            self.model.learn(total_timesteps=REALLY_BIG_INT,  # we'll stop manually
                             callback=self._training_callback)
        except KeyboardInterrupt:
            pass

        # save and evaluate final model, if not done yet
        if len(self.win_rates) < self.num_evals:
            self._training_callback()

        # close the envs
        for e in (self.env, self.evaluator):
            e.close()


class SelfPlay(TrainingSession):
    def __init__(self, model_builder, model_params, env_builder, env_params, eval_env_params,
                 train_episodes, eval_episodes, num_evals, switch_frequency, path,
                 seed, num_envs=1):
        super(SelfPlay, self).__init__(model_params, path, seed)

        # log start time
        start_time = time.perf_counter()

        # initialize parallel training environments
        self.logger.debug("Initializing training envs...")
        env = []

        for i in range(num_envs):
            # no overlap between episodes at each process
            if seed is not None:
                current_seed = seed + (train_episodes // num_envs) * i
            else:
                current_seed = None

            # create one env per process
            env.append(lambda: env_builder(**env_params))

        # wrap envs in a vectorized env
        self.env = DummyVecEnv(env)

        # initialize parallel evaluating environments
        self.logger.debug("Initializing evaluation envs...")
        eval_seed = seed + train_episodes if seed is not None else None
        self.evaluator: Evaluator = Evaluator(env_builder, eval_env_params,
                                              eval_episodes // 2, eval_seed, num_envs)

        # build the models
        self.logger.debug("Building the models...")
        self.model = model_builder(self.env, seed, **model_params)
        self.model.adversary = model_builder(self.env, seed, **model_params)

        # initialize parameters of adversary models accordingly
        self.model.adversary.load_parameters(self.model.get_parameters(), exact_match=True)

        # set adversary models as adversary policies of the self-play envs
        def make_adversary_policy(model, env):
            def adversary_policy(obs):
                zero_completed_obs = np.zeros((num_envs,) + env.observation_space.shape)
                zero_completed_obs[0, :] = obs

                actions, _ = model.adversary.predict(zero_completed_obs)

                return actions[0]

            return adversary_policy

        self.env.set_attr('adversary_policy',
                          make_adversary_policy(self.model, self.env))

        # create necessary folders
        os.makedirs(self.path, exist_ok=True)

        # set tensorflow log dirs
        self.model.tensorflow_log = self.path

        # save parameters
        self.train_episodes = train_episodes
        self.eval_episodes = eval_episodes
        self.num_evals = num_evals
        self.switch_frequency = switch_frequency
        self.eval_frequency = train_episodes / num_evals
        self.num_switches = math.ceil(train_episodes / switch_frequency)

        # initialize control attributes
        self.model.last_eval, self.model.next_eval = None, 0
        self.model.last_switch, self.model.next_switch = None, self.switch_frequency

        # initialize results
        self.checkpoints = []
        self.win_rates = []
        self.episode_lengths = []
        self.action_histograms = []

        # log end time
        end_time = time.perf_counter()

        self.logger.debug("Finished initializing training session "
                          f"({round(end_time - start_time, ndigits=3)}s).")

    def _training_callback(self, _locals=None, _globals=None):
        model = _locals['self']
        episodes_so_far = model.num_timesteps // 30

        turns = model.env.get_attr('turn')
        playing_first = model.env.get_attr('play_first')

        for i in range(model.env.num_envs):
            if turns[i] in range(0, model.env.num_envs):
                model.env.set_attr('play_first', not playing_first[i], indices=[i])

        # if it is time to evaluate, do so
        if episodes_so_far >= model.next_eval:
            # save model
            model_path = self.path + f'/{episodes_so_far}'
            model.save(model_path)
            save_model_as_json(model, self.params['activation'], model_path)
            self.logger.debug(f"Saved model at {model_path}.zip/json.")

            # evaluate the model
            self.logger.info(f"Evaluating model ({episodes_so_far} episodes)...")
            start_time = time.perf_counter()

            if self.evaluator.seed is not None:
                self.evaluator.seed = self.seed + self.train_episodes
            mean_reward, ep_length, act_hist = \
                self.evaluator.run(model, play_first=True)

            if self.evaluator.seed is not None:
                self.evaluator.seed += self.eval_episodes
            mean_reward2, ep_length2, act_hist2 = \
                self.evaluator.run(model, play_first=False)

            mean_reward = (mean_reward + mean_reward2) / 2
            ep_length = (ep_length + ep_length2) / 2
            act_hist = [(act_hist[i] + act_hist2[i]) / 2 for i in range(3)]

            end_time = time.perf_counter()
            self.logger.info(f"Finished evaluating "
                             f"({round(end_time - start_time, 3)}s). "
                             f"Avg. reward: {mean_reward}")

            # save the results
            self.checkpoints.append(episodes_so_far)
            self.win_rates.append((mean_reward + 1) / 2)
            self.episode_lengths.append(ep_length)
            self.action_histograms.append(act_hist)

            # update control attributes
            model.last_eval = episodes_so_far
            model.next_eval += self.eval_frequency

            # write partial results to file
            self._save_results()

        # if training should end, return False to end training
        training_is_finished = episodes_so_far >= model.next_switch

        if training_is_finished:
            model.last_switch = episodes_so_far
            model.next_switch += self.switch_frequency

        return not training_is_finished

    def _train(self):
        # save and evaluate starting models
        self._training_callback({'self': self.model})

        try:
            self.logger.debug(f"Training will switch models every "
                              f"{self.switch_frequency} episodes")

            for _ in range(self.num_switches):
                # train the model
                self.model.learn(total_timesteps=REALLY_BIG_INT,
                                 reset_num_timesteps=False,
                                 callback=self._training_callback)
                self.logger.debug(f"Model trained for "
                                  f"{self.model.num_timesteps // 30} episodes. ")

                # update parameters of adversary models
                self.model.adversary.load_parameters(self.model.get_parameters(),
                                                     exact_match=True)
                self.logger.debug("Parameters of adversary network updated.")
        except KeyboardInterrupt:
            pass

        self.logger.debug(f"Training ended at {self.model.num_timesteps // 30} "
                          f"episodes")

        # save and evaluate final models, if not done yet
        if len(self.win_rates) < self.num_evals:
            self._training_callback({'self': self.model})

        if len(self.win_rates) < self.num_evals:
            self._training_callback({'self': self.model})

        # close the envs
        for e in (self.env, self.evaluator):
            e.close()


class AsymmetricSelfPlay(TrainingSession):
    def __init__(self, model_builder, model_params, env_builder, env_params, eval_env_params,
                 train_episodes, eval_episodes, num_evals,
                 switch_frequency, path, seed, num_envs=1):
        super(AsymmetricSelfPlay, self).__init__(model_params, path, seed)

        # log start time
        start_time = time.perf_counter()

        # initialize parallel training environments
        self.logger.debug("Initializing training envs...")
        env1, env2 = [], []

        for i in range(num_envs):
            # no overlap between episodes at each process
            if seed is not None:
                current_seed = seed + (train_episodes // num_envs) * i
            else:
                current_seed = None

            # create one env per process
            env1.append(lambda: env_builder(**env_params))
            env2.append(lambda: env_builder(**env_params))

        # wrap envs in a vectorized env
        self.env1 = DummyVecEnv(env1)
        self.env2 = DummyVecEnv(env2)

        # initialize parallel evaluating environments
        self.logger.debug("Initializing evaluation envs...")
        eval_seed = seed + train_episodes if seed is not None else None
        self.evaluator: Evaluator = Evaluator(env_builder, eval_env_params,
                                              eval_episodes, eval_seed, num_envs)

        # build the models
        self.logger.debug("Building the models...")
        self.model1 = model_builder(self.env1, seed, **model_params)
        self.model1.adversary = model_builder(self.env2, seed, **model_params)
        self.model2 = model_builder(self.env2, seed, **model_params)
        self.model2.adversary = model_builder(self.env1, seed, **model_params)

        # initialize parameters of adversary models accordingly
        self.model1.adversary.load_parameters(self.model2.get_parameters(), exact_match=True)
        self.model2.adversary.load_parameters(self.model1.get_parameters(), exact_match=True)

        # set adversary models as adversary policies of the self-play envs
        def make_adversary_policy(model, env):
            def adversary_policy(obs):
                zero_completed_obs = np.zeros((num_envs,) + env.observation_space.shape)
                zero_completed_obs[0, :] = obs

                actions, _ = model.adversary.predict(zero_completed_obs)

                return actions[0]

            return adversary_policy

        self.env1.set_attr('adversary_policy',
                           make_adversary_policy(self.model1, self.env1))
        self.env2.set_attr('adversary_policy',
                           make_adversary_policy(self.model2, self.env2))

        # create necessary folders
        os.makedirs(self.path + '/role0', exist_ok=True)
        os.makedirs(self.path + '/role1', exist_ok=True)

        # set tensorflow log dirs
        self.model1.tensorflow_log = self.path + '/role0'
        self.model2.tensorflow_log = self.path + '/role1'

        # save parameters
        self.train_episodes = train_episodes
        self.eval_episodes = eval_episodes
        self.num_evals = num_evals
        self.switch_frequency = switch_frequency
        self.eval_frequency = train_episodes / num_evals
        self.num_switches = math.ceil(train_episodes / switch_frequency)

        # initialize control attributes
        self.model1.role_id, self.model2.role_id = 0, 1
        self.model1.last_eval, self.model1.next_eval = None, 0
        self.model2.last_eval, self.model2.next_eval = None, 0
        self.model1.last_switch, self.model1.next_switch = 0, self.switch_frequency
        self.model2.last_switch, self.model2.next_switch = 0, self.switch_frequency

        # initialize results
        self.checkpoints = [], []
        self.win_rates = [], []
        self.episode_lengths = [], []
        self.action_histograms = [], []

        # log end time
        end_time = time.perf_counter()

        self.logger.debug("Finished initializing training session "
                          f"({round(end_time - start_time, ndigits=3)}s).")

    def _training_callback(self, _locals=None, _globals=None):
        model = _locals['self']
        episodes_so_far = model.num_timesteps // 30

        # if it is time to evaluate, do so
        if episodes_so_far >= model.next_eval:
            # save model
            model_path = f'{self.path}/role{model.role_id}/{episodes_so_far}'
            model.save(model_path)
            save_model_as_json(model, self.params['activation'], model_path)
            self.logger.debug(f"Saved model at {model_path}.zip/json.")

            # evaluate the model
            self.logger.info(f"Evaluating model {model.role_id} "
                             f"({episodes_so_far} episodes)...")
            start_time = time.perf_counter()

            mean_reward, ep_length, act_hist = \
                self.evaluator.run(model, play_first=model.role_id == 0)

            end_time = time.perf_counter()
            self.logger.info(f"Finished evaluating "
                             f"({round(end_time - start_time, 3)}s). "
                             f"Avg. reward: {mean_reward}")

            # save the results
            self.checkpoints[model.role_id].append(episodes_so_far)
            self.win_rates[model.role_id].append((mean_reward + 1) / 2)
            self.episode_lengths[model.role_id].append(ep_length)
            self.action_histograms[model.role_id].append(act_hist)

            # update control attributes
            model.last_eval = episodes_so_far
            model.next_eval += self.eval_frequency

            # write partial results to file
            self._save_results()

        # if training should end, return False to end training
        training_is_finished = episodes_so_far >= model.next_switch

        if training_is_finished:
            model.last_switch = episodes_so_far
            model.next_switch += self.switch_frequency

        return not training_is_finished

    def _train(self):
        # save and evaluate starting models
        self._training_callback({'self': self.model1})
        self._training_callback({'self': self.model2})

        try:
            self.logger.debug(f"Training will switch models every "
                              f"{self.switch_frequency} episodes")

            for _ in range(self.num_switches):
                # train the first player model
                self.model1.learn(total_timesteps=REALLY_BIG_INT,
                                  reset_num_timesteps=False,
                                  callback=self._training_callback)
                self.logger.debug(f"Model {self.model1.role_id} trained for "
                                  f"{self.model1.num_timesteps // 30} episodes. "
                                  f"Switching to model {self.model2.role_id}.")

                # train the second player model
                self.model2.learn(total_timesteps=REALLY_BIG_INT,
                                  reset_num_timesteps=False,
                                  callback=self._training_callback)
                self.logger.debug(f"Model {self.model2.role_id} trained for "
                                  f"{self.model2.num_timesteps // 30} episodes. "
                                  f"Switching to model {self.model1.role_id}.")

                # update parameters of adversary models
                self.model1.adversary.load_parameters(self.model2.get_parameters(),
                                                      exact_match=True)
                self.model2.adversary.load_parameters(self.model1.get_parameters(),
                                                      exact_match=True)
                self.logger.debug("Parameters of adversary networks updated.")
        except KeyboardInterrupt:
            pass

        self.logger.debug(f"Training ended at {self.model1.num_timesteps // 30} "
                          f"episodes")

        # save and evaluate final models, if not done yet
        if len(self.win_rates[0]) < self.num_evals:
            self._training_callback({'self': self.model1})

        if len(self.win_rates[1]) < self.num_evals:
            self._training_callback({'self': self.model1})

        # close the envs
        for e in (self.env1, self.env2, self.evaluator):
            e.close()


class Evaluator:
    def __init__(self, env_builder, env_params, episodes, seed, num_envs):
        # log start time
        start_time = time.perf_counter()

        # initialize logger
        self.logger = logging.getLogger('{0}.{1}'.format(__name__, type(self).__name__))

        # initialize parallel environments
        self.logger.debug("Initializing envs...")
        self.env = [lambda: env_builder(**env_params)
                    for _ in range(num_envs)]
        self.env: VecEnv = DummyVecEnv(self.env)

        # save parameters
        self.episodes = episodes
        self.seed = seed

        # log end time
        end_time = time.perf_counter()

        self.logger.debug("Finished initializing evaluator "
                          f"({round(end_time - start_time, ndigits=3)}s).")

    def run(self, agent: PPO, play_first=True):
        """
        Evaluates an agent.
        :param agent: (gym_locm.agents.Agent) Agent to be evaluated.
        :param play_first: Whether the agent will be playing first.
        :return: A tuple containing the `mean_reward`, the `mean_length`
        and the `action_histogram` of the evaluation episodes.
        """
        # set appropriate seeds
        if self.seed is not None:
            for i in range(self.env.num_envs):
                current_seed = self.seed
                current_seed += (self.episodes // self.env.num_envs) * i
                current_seed -= 1  # resetting the env increases the seed by one

                self.env.env_method('seed', current_seed, indices=[i])

        # set agent role
        self.env.set_attr('play_first', play_first)

        # reset the env
        observations = self.env.reset()

        # initialize metrics
        episodes_so_far = 0
        episode_rewards = [[0.0] for _ in range(self.env.num_envs)]
        episode_lengths = [[0] for _ in range(self.env.num_envs)]
        action_histogram = [0] * self.env.action_space.n

        # run the episodes
        while True:
            # get the agent's action for all parallel envs
            actions, _ = agent.predict(observations, deterministic=True)

            # update the action histogram
            for action in actions:
                action_histogram[action] += 1

            # perform the action and get the outcome
            observations, rewards, dones, _ = self.env.step(actions)

            # update metrics
            for i in range(self.env.num_envs):
                episode_rewards[i][-1] += rewards[i]
                episode_lengths[i][-1] += 1

                if dones[i]:
                    episode_rewards[i].append(0.0)
                    episode_lengths[i].append(0)

                    episodes_so_far += 1

            # check exiting condition
            if episodes_so_far >= self.episodes:
                break

        # join all parallel metrics
        all_rewards = [reward for rewards in episode_rewards
                       for reward in rewards[:-1]]
        all_lengths = [length for lengths in episode_lengths
                       for length in lengths[:-1]]

        # transform the action histogram in a probability distribution
        action_histogram = [action_freq / sum(action_histogram)
                            for action_freq in action_histogram]

        # cap any unsolicited additional episodes
        all_rewards = all_rewards[:self.episodes]
        all_lengths = all_lengths[:self.episodes]

        return mean(all_rewards), mean(all_lengths), action_histogram

    def close(self):
        self.env.close()


def save_model_as_json(model, act_fun, path):
    pass
    '''with open(path + '.json', 'w') as json_file:
        params = {}

        # create a parameter dictionary
        for label, weights in model.get_parameters().items():
            params[label] = weights.tolist()

        # add activation function to it
        params['act_fun'] = act_fun

        # and save into the new file
        json.dump(params, json_file)'''


def model_builder_mlp(env, seed, neurons, layers, activation, n_steps, nminibatches,
                      noptepochs, cliprange, vf_coef, ent_coef, learning_rate):
    net_arch = [neurons] * layers
    activation = dict(tanh=Tanh, relu=ReLU, elu=ELU)[activation]

    return PPO(CnnPolicy, env, verbose=0, gamma=1, seed=seed,
               policy_kwargs=dict(
                   features_extractor_class=CustomCNN,
                   features_extractor_kwargs=dict(features_dim=256)),
               n_steps=n_steps, batch_size=nminibatches,
               n_epochs=noptepochs, clip_range=cliprange,
               vf_coef=vf_coef, ent_coef=ent_coef, learning_rate=learning_rate)


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = th.nn.Sequential(
            th.nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=2),
            th.nn.ReLU(),
            th.nn.Conv2d(16, 32, kernel_size=2),
            th.nn.ReLU(),
            th.nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = th.nn.Sequential(
            th.nn.Linear(n_flatten, features_dim),
            th.nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


if __name__ == '__main__':
    '''env_params = {
        'battle_agents': (MaxAttackBattleAgent(), MaxAttackBattleAgent()),
        'use_draft_history': False,
        'use_mana_curve': False
    }

    eval_env_params = {
        'draft_agent': MaxAttackDraftAgent(),
        'battle_agents': (MaxAttackBattleAgent(), MaxAttackBattleAgent()),
        'use_draft_history': False,
        'use_mana_curve': False
    }

    model_params = {'layers': 1, 'neurons': 29, 'n_steps': 30, 'nminibatches': 30,
                    'noptepochs': 19, 'cliprange': 0.1, 'vf_coef': 1.0,
                    'ent_coef': 0.00781891437626065, 'activation': 'tanh',
                    'learning_rate': 0.0001488768154153614}

    ts = FixedAdversary(model_builder_mlp, model_params, env_params, eval_env_params,
                        30000, 1000, 12, True, 'models/trashcan/trash04', 36987,
                        num_envs=4)

    ts.run()'''
