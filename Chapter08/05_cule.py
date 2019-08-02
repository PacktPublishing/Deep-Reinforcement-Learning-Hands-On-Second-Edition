import torch
from torchcule.atari import Env


if __name__ == "__main__":
    e = Env('PongNoFrameskip-v4', 2, color_mode='gray',
            device=torch.device('cuda', 0), rescale=True, clip_rewards=True,
            episodic_life=True, repeat_prob=0.0)
    obs = e.reset(initial_steps=4000, verbose=False)
    print(obs)
