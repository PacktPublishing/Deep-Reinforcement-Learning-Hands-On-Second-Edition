import gym
import logging
from typing import Optional, Iterable, List
from textworld.gym import spaces as tw_spaces

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class TextWorldPreproc(gym.Wrapper):
    """
    Simple wrapper to preprocess text_world game observation

    Observation and action spaces are not handled, as it will
    be wrapped into other preprocessors
    """
    log = logging.getLogger("TextWorldPreproc")

    def __init__(self, env: gym.Env, encode_raw_text: bool = False,
                 encode_extra_fields: Iterable[str] = (
                         'description', 'inventory'),
                 use_admissible_commands: bool = True,
                 keep_admissible_commands: bool = False,
                 use_intermediate_reward: bool = True,
                 tokens_limit: Optional[int] = None,
                 reward_wrong_last_command: Optional[float] = None):
        """
        :param env: TextWorld env to be wrapped
        :param encode_raw_text: flag to encode raw texts
        :param encode_extra_fields: fields to be encoded
        :param use_admissible_commands: use list of commands
        :param keep_admissible_commands: keep list of admissible commands in observations
        :param use_intermediate_reward: intermediate reward
        :param tokens_limit: limit tokens in encoded fields
        :param reward_wrong_last_command: if given, this reward will be given if 'last_command' observation field is 'None'.
        """
        super(TextWorldPreproc, self).__init__(env)
        if not isinstance(env.observation_space, tw_spaces.Word):
            raise ValueError(
                "Env should expose textworld obs, "
                "got %s instead" % env.observation_space)
        self._encode_raw_text = encode_raw_text
        self._encode_extra_field = tuple(encode_extra_fields)
        self._use_admissible_commands = use_admissible_commands
        self._keep_admissible_commands = keep_admissible_commands
        self._use_intermedate_reward = use_intermediate_reward
        self._num_fields = len(self._encode_extra_field) + \
                           int(self._encode_raw_text)
        self._last_admissible_commands = None
        self._last_extra_info = None
        self._tokens_limit = tokens_limit
        self._reward_wrong_last_command = reward_wrong_last_command
        self._cmd_hist = []

    @property
    def num_fields(self):
        return self._num_fields

    def _encode(self, obs: str, extra_info: dict) -> dict:
        obs_result = []
        if self._encode_raw_text:
            tokens = self.env.observation_space.tokenize(obs)
            if self._tokens_limit is not None:
                tokens = tokens[:self._tokens_limit]
            obs_result.append(tokens)
        for field in self._encode_extra_field:
            extra = extra_info[field]
            tokens = self.env.observation_space.tokenize(extra)
            if self._tokens_limit is not None:
                tokens = tokens[:self._tokens_limit]
            obs_result.append(tokens)
        result = {"obs": obs_result}
        if self._use_admissible_commands:
            adm_result = []
            for cmd in extra_info['admissible_commands']:
                cmd_tokens = self.env.action_space.tokenize(cmd)
                adm_result.append(cmd_tokens)
            result['admissible_commands'] = adm_result
            self._last_admissible_commands = \
                extra_info['admissible_commands']
        if self._keep_admissible_commands:
            result['admissible_commands'] = extra_info['admissible_commands']
            if 'policy_commands' in extra_info:
                result['policy_commands'] = extra_info['policy_commands']
        self._last_extra_info = extra_info
        return result

    # TextWorld environment has a workaround of gym drawback:
    # reset returns tuple with raw observation and extra dict
    def reset(self):
        res = self.env.reset()
        self._cmd_hist = []
        return self._encode(res[0], res[1])

    def step(self, action):
        if self._use_admissible_commands:
            action = self._last_admissible_commands[action]
            self._cmd_hist.append(action)
        obs, r, is_done, extra = self.env.step(action)
        if self._use_intermedate_reward:
            r += extra.get('intermediate_reward', 0)
        if self._reward_wrong_last_command is not None:
            if action not in self._last_extra_info['admissible_commands']:
                r += self._reward_wrong_last_command
        new_extra = dict(extra)
        fields = list(self._encode_extra_field)
        fields.append('admissible_commands')
        fields.append('intermediate_reward')
        for f in fields:
            if f in new_extra:
                new_extra.pop(f)
        return self._encode(obs, extra), r, is_done, new_extra

    @property
    def last_admissible_commands(self):
        if self._last_admissible_commands:
            return tuple(self._last_admissible_commands)
        return None

    @property
    def last_extra_info(self):
        return self._last_extra_info


class Encoder(nn.Module):
    """
    Takes input sequences (after embeddings) and returns
    the hidden state from LSTM
    """
    def __init__(self, emb_size: int, out_size: int):
        super(Encoder, self).__init__()

        self.net = nn.LSTM(
            input_size=emb_size, hidden_size=out_size,
            batch_first=True)

    def forward(self, x):
        self.net.flatten_parameters()
        _, hid_cell = self.net(x)
        # Warn: if bidir=True or several layers,
        # sequeeze has to be changed!
        return hid_cell[0].squeeze(0)


class Preprocessor(nn.Module):
    """
    Takes batch of several input sequences and outputs their
    summary from one or many encoders
    """
    def __init__(self, dict_size: int, emb_size: int,
                 num_sequences: int, enc_output_size: int):
        """
        :param dict_size: amount of words is our vocabulary
        :param emb_size: dimensionality of embeddings
        :param num_sequences: count of sequences
        :param enc_output_size: output from single encoder
        """
        super(Preprocessor, self).__init__()

        self._enc_output_size = enc_output_size
        self.emb = nn.Embedding(num_embeddings=dict_size,
                                embedding_dim=emb_size)
        self.encoders = []
        for idx in range(num_sequences):
            enc = Encoder(emb_size, enc_output_size)
            self.encoders.append(enc)
            self.add_module(f"enc_{idx}", enc)
        self.enc_commands = Encoder(emb_size, enc_output_size)

    @property
    def obs_enc_size(self):
        return self._enc_output_size * len(self.encoders)

    @property
    def cmd_enc_size(self):
        return self._enc_output_size

    def _apply_encoder(self, batch: List[List[int]],
                       encoder: Encoder):
        dev = self.emb.weight.device
        batch_t = [self.emb(torch.tensor(sample).to(dev))
                   for sample in batch]
        batch_seq = rnn_utils.pack_sequence(
            batch_t, enforce_sorted=False)
        return encoder(batch_seq)

    def encode_sequences(self, batches):
        """
        Forward pass of Preprocessor
        :param batches: list of tuples with variable-length sequences of word ids
        :return: tensor with concatenated encoder outputs for every batch sample
        """
        data = []
        for enc, enc_batch in zip(self.encoders, zip(*batches)):
            data.append(self._apply_encoder(enc_batch, enc))
        res_t = torch.cat(data, dim=1)
        return res_t

    def encode_commands(self, batch):
        """
        Apply encoder to list of commands sequence
        :param batch: list of lists of idx
        :return: tensor with encoded commands in original order
        """
        return self._apply_encoder(batch, self.enc_commands)
