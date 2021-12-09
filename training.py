import argparse
import os
import sys

import numpy as np
import wandb

from gym_microrts.envs.vec_env import MicroRTSScriptEnv
from experiments.scripts.trainer import FixedAdversary, model_builder_mlp

_counter = 0


def get_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--opponent", "-o", default="coacAI")
    p.add_argument("--path", "-p", help="path to save models and results",
                   required=True)

    p.add_argument("--train-episodes", "-te", help="how many episodes to train",
                   default=30000, type=int)
    p.add_argument("--eval-episodes", "-ee", help="how many episodes to eval",
                   default=1000, type=int)
    p.add_argument("--num-evals", "-ne", type=int, default=12,
                   help="how many evaluations to perform throughout training")

    p.add_argument("--switch-freq", type=int, default=1000,
                   help="how many episodes to run before updating opponent networks")
    p.add_argument("--layers", type=int, default=1,
                   help="amount of layers in the network")
    p.add_argument("--neurons", type=int, default=169,
                   help="amount of neurons on each hidden layer in the network")
    p.add_argument("--act-fun", choices=['tanh', 'relu', 'elu'], default='elu',
                   help="activation function of neurons in hidden layers")
    p.add_argument("--n-steps", type=int, default=270,
                   help="batch size (in timesteps, 30 timesteps = 1 episode)")
    p.add_argument("--nminibatches", type=int, default=135,
                   help="amount of minibatches created from the batch")
    p.add_argument("--noptepochs", type=int, default=20,
                   help="amount of epochs to train with all minibatches")
    p.add_argument("--cliprange", type=float, default=0.1,
                   help="clipping range of the loss function")
    p.add_argument("--vf-coef", type=float, default=1.0,
                   help="weight of the value function in the loss function")
    p.add_argument("--ent-coef", type=float, default=0.00595,
                   help="weight of the entropy term in the loss function")
    p.add_argument("--learning-rate", type=float, default=0.000228,
                   help="learning rate")

    p.add_argument("--seed", type=int, default=None,
                   help="seed to use on the model, envs and training")
    p.add_argument("--concurrency", type=int, default=1,
                   help="amount of environments to use")

    return p


def run():
    if sys.version_info < (3, 0, 0):
        sys.stderr.write("You need python 3.0 or later to run this script\n")
        sys.exit(1)

    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    os.makedirs(args.path, exist_ok=True)

    env_builder = lambda **kwargs: MicroRTSScriptEnv(**kwargs)

    env_params = {
        'ai2': args.opponent,
        'max_steps': 2000,
        'render_theme': 2,
        'map_path': 'maps/16x16/basesWorkers16x16.xml',
        'reward_weight': np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
    }

    eval_env_params = env_params

    model_params = {'layers': args.layers, 'neurons': args.neurons,
                    'n_steps': args.n_steps, 'nminibatches': args.nminibatches,
                    'noptepochs': args.noptepochs, 'cliprange': args.cliprange,
                    'vf_coef': args.vf_coef, 'ent_coef': args.ent_coef,
                    'activation': args.act_fun, 'learning_rate': args.learning_rate,
                    'tensorboard_log': args.path + '/tf_logs'}

    run = wandb.init(
        project="gym-microrts-scripts",
        entity="ronaldosvieira",
        sync_tensorboard=True,
        config=args
    )

    trainer = FixedAdversary(model_builder_mlp, model_params, env_builder,
                             env_params, eval_env_params, args.train_episodes,
                             args.eval_episodes, args.num_evals,
                             True, args.path, args.seed, args.concurrency,
                             wandb_run=run)

    trainer.run()

    run.finish()


if __name__ == "__main__":
    run()
