import numpy as np
import pickle
import torch
from docopt import docopt
from model import ActorCriticModel
from utils import create_env

from smarts.core.utils.episodes import episodes

def main():
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        enjoy.py [options]
        enjoy.py --help
    
    Options:
        --model=<path>              Specifies the path to the trained model [default: ./models/run.nn].
        --num_episodes=<str>        Number of episodes [default: 5].
        --scenario=<path>           Specifies scenario [default: ./scenarios/roundabout].
        --envision                  envision [default: False].
        --sumo                      sumo [default: False].
        --visdom                    visdom [default: False].
    """
    options = docopt(_USAGE)
    model_path = options["--model"]
    num_episodes = int(options["--num_episodes"])
    scenario_path = [options["--scenario"]]
    envision = options["--envision"]
    sumo = options["--sumo"]
    visdom = options["--visdom"]

    print(model_path)
    print(envision, sumo, visdom)

    # Inference device
    device = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")

    # Load model and config
    state_dict, config = pickle.load(open(model_path, "rb"))

    # Instantiate environment
    env = create_env(config["env"],
                    scenario_path=scenario_path,
                    envision=envision,
                    visdom=visdom,
                    sumo=sumo)

    # Initialize model and load its parameters
    model = ActorCriticModel(config, env.observation_space, (env.action_space.n,))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Run and render episode
    done = False

    # Init recurrent cell
    hxs, cxs = model.init_recurrent_cell_states(1, device)
    if config["recurrence"]["layer_type"] == "gru":
        recurrent_cell = hxs
    elif config["recurrence"]["layer_type"] == "lstm":
        recurrent_cell = (hxs, cxs)

    obs = env.reset()

    for episode in episodes(n=num_episodes):
        obs = env.reset()
        episode.record_scenario(env._env.scenario_log)

        done = False

        while not done:
            # Render environment
            env.render()
            # Forward model
            policy, value, recurrent_cell = model(torch.tensor(np.expand_dims(obs, 0)), recurrent_cell, device, 1)
            # Sample action
            action = policy.sample().cpu().numpy()
            # Step environemnt
            obs, reward, done, info = env._env.step({env._env.agent_id: action})

            episode.record_step(obs, reward, done, info)

            done = done['__all__']
            obs = obs[env.AGENT_ID]
    
    # after done, render last state
    env.render()

    env.close()

if __name__ == "__main__":
    main()