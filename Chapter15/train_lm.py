import gym
import ptan
from textworld.gym import register_games
from textworld.envs.wrappers.filter import EnvInfos

from lib import preproc, model, common

GAMMA = 0.9


EXTRA_GAME_INFO = {
    "inventory": True,
    "description": True,
    "intermediate_reward": True,
    "last_command": True,
}


def run(device = "cpu"):
    env_id = register_games(["games/simple1.ulx"], request_infos=EnvInfos(**EXTRA_GAME_INFO))
    env = gym.make(env_id)
    env = preproc.TextWorldPreproc(env, use_admissible_commands=False, reward_wrong_last_command=-1)
    params = common.PARAMS['small']

    prep = preproc.Preprocessor(
        dict_size=env.observation_space.vocab_size,
        emb_size=params.embeddings, num_sequences=env.num_fields,
        enc_output_size=params.encoder_size)

    cmd = model.CommandModel(prep.obs_enc_size, env.observation_space.vocab_size, prep.emb,
                             max_tokens=env.action_space.max_length,
                             max_commands=5, start_token=env.action_space.BOS_id,
                             sep_token=env.action_space.SEP_id)
    net = model.A2CModel(obs_size=env.num_fields * params.encoder_size)
    agent = model.CmdAgent(env, cmd, prep, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=GAMMA, steps_count=1)

    batch = []
    for exp in exp_source:
        batch.append(exp)
        break

    s = env.reset()
    obs_t = prep.encode_sequences([s['obs']])
    print(obs_t)
    tokens, logits = cmd(obs_t)

    return env, prep, cmd


if __name__ == "__main__":
    run()
