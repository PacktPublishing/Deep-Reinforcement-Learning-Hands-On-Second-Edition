import enum
import collections

from . import _env
from . import _common

State = collections.namedtuple("State", field_names=['corner_pos', 'corner_ort'])
RenderedState = collections.namedtuple("RenderedState", field_names=['top', 'front', 'left',
                                                                     'right', 'back', 'bottom'])

initial_state = State(corner_pos=tuple(range(8)), corner_ort=tuple([0]*8))


def is_initial(state):
    assert isinstance(state, State)
    return state.corner_pos == initial_state.corner_pos and \
           state.corner_ort == initial_state.corner_ort


# available actions. Capital actions denote clockwise rotation
class Action(enum.Enum):
    R = 0
    L = 1
    T = 2
    D = 3
    F = 4
    B = 5
    r = 6
    l = 7
    t = 8
    d = 9
    f = 10
    b = 11


_inverse_action = {
    Action.R: Action.r,
    Action.r: Action.R,
    Action.L: Action.l,
    Action.l: Action.L,
    Action.T: Action.t,
    Action.t: Action.T,
    Action.D: Action.d,
    Action.d: Action.D,
    Action.F: Action.f,
    Action.f: Action.F,
    Action.B: Action.b,
    Action.b: Action.B
}


def inverse_action(action):
    assert isinstance(action, Action)
    return _inverse_action[action]


_transform_map = {
    Action.R: [
        ((1, 2), (2, 6), (6, 5), (5, 1)),           # corner map
        ((1, 2), (2, 1), (5, 1), (6, 2)),           # corner rotate
    ],
    Action.L: [
        ((3, 0), (7, 3), (0, 4), (4, 7)),
        ((0, 1), (3, 2), (4, 2), (7, 1)),
    ],
    Action.T: [
        ((0, 3), (1, 0), (2, 1), (3, 2)),
        (),
    ],
    Action.D: [
        ((4, 5), (5,  6), (6, 7), (7, 4)),
        (),
    ],
    Action.F: [
        ((0, 1), (1, 5), (5, 4), (4, 0)),
        ((0, 2), (1, 1), (4, 1), (5, 2)),
    ],
    Action.B: [
        ((2, 3), (3, 7), (7, 6), (6, 2)),
        ((2, 2), (3, 1), (6, 1), (7, 2)),
    ]
}


def transform(state, action):
    assert isinstance(state, State)
    assert isinstance(action, Action)
    global _transform_map

    is_inv = action not in _transform_map
    if is_inv:
        action = inverse_action(action)
    c_map, c_rot = _transform_map[action]
    corner_pos = _common._permute(state.corner_pos, c_map, is_inv)
    corner_ort = _common._permute(state.corner_ort, c_map, is_inv)
    corner_ort = _common._rotate(corner_ort, c_rot)
    return State(corner_pos=tuple(corner_pos), corner_ort=tuple(corner_ort))


# create initial sides in the right order
def _init_sides():
    return [
        [None for _ in range(4)]
        for _ in range(6)               # top, left, back, front, right, bottom
    ]


# corner cubelets colors (clockwise from main label). Order of cubelets are first top,
# in counter-clockwise, started from front left
corner_colors = (
    ('W', 'R', 'G'), ('W', 'B', 'R'), ('W', 'O', 'B'), ('W', 'G', 'O'),
    ('Y', 'G', 'R'), ('Y', 'R', 'B'), ('Y', 'B', 'O'), ('Y', 'O', 'G')
)


# map every 3-side cubelet to their projection on sides
# sides are indexed in the order of _init_sides() function result
corner_maps = (
    # top layer
    ((0, 2), (3, 0), (1, 1)),
    ((0, 3), (4, 0), (3, 1)),
    ((0, 1), (2, 0), (4, 1)),
    ((0, 0), (1, 0), (2, 1)),
    # bottom layer
    ((5, 0), (1, 3), (3, 2)),
    ((5, 1), (3, 3), (4, 2)),
    ((5, 3), (4, 3), (2, 2)),
    ((5, 2), (2, 3), (1, 2))
)


# render state into human readable form
def render(state):
    assert isinstance(state, State)
    global corner_colors, corner_maps

    sides = _init_sides()

    for corner, orient, maps in zip(state.corner_pos, state.corner_ort, corner_maps):
        cols = corner_colors[corner]
        cols = _common._map_orient(cols, orient)
        for (arr_idx, index), col in zip(maps, cols):
            sides[arr_idx][index] = col

    return RenderedState(top=sides[0], left=sides[1], back=sides[2], front=sides[3],
                         right=sides[4], bottom=sides[5])


encoded_shape = (8, 24)


def encode_inplace(target, state):
    """
    Encode cude into existig zeroed numpy array
    Follows encoding described in paper https://arxiv.org/abs/1805.07470
    :param target: numpy array
    :param state: state to be encoded
    """
    assert isinstance(state, State)

    # handle corner cubelets: find their permuted position
    for corner_idx in range(8):
        perm_pos = state.corner_pos.index(corner_idx)
        corn_ort = state.corner_ort[perm_pos]
        target[corner_idx, perm_pos * 3 + corn_ort] = 1


# register env
_env.register(_env.CubeEnv(name="cube2x2", state_type=State, initial_state=initial_state,
                           is_goal_pred=is_initial, action_enum=Action,
                           transform_func=transform, inverse_action_func=inverse_action,
                           render_func=render, encoded_shape=encoded_shape, encode_func=encode_inplace))
