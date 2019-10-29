import unittest
import numpy as np
import random

from libcube.cubes import cube2x2


class CubeRender(unittest.TestCase):
    def test_init_render(self):
        state = cube2x2.initial_state
        render = cube2x2.render(state)
        self.assertIsInstance(render, cube2x2.RenderedState)
        self.assertEqual(render.top, ['W'] * 4)
        self.assertEqual(render.back, ['O'] * 4)
        self.assertEqual(render.bottom, ['Y'] * 4)
        self.assertEqual(render.front, ['R'] * 4)
        self.assertEqual(render.left, ['G'] * 4)
        self.assertEqual(render.right, ['B'] * 4)


class CubeTransforms(unittest.TestCase):
    def test_top(self):
        s = cube2x2.initial_state
        s = cube2x2.transform(s, cube2x2.Action.T)
        r = cube2x2.render(s)
        self.assertEqual(r.top, ['W'] * 4)
        self.assertEqual(r.back, ['G'] * 2 + ['O'] * 2)
        self.assertEqual(r.bottom, ['Y'] * 4)
        self.assertEqual(r.front, ['B'] * 2 + ['R'] * 2)
        self.assertEqual(r.left, ['R'] * 2 + ['G'] * 2)
        self.assertEqual(r.right, ['O'] * 2 + ['B'] * 2)

    def test_top_rev(self):
        s = cube2x2.initial_state
        s = cube2x2.transform(s, cube2x2.Action.t)
        r = cube2x2.render(s)
        self.assertEqual(r.top, ['W'] * 4)
        self.assertEqual(r.back, ['B'] * 2 + ['O'] * 2)
        self.assertEqual(r.bottom, ['Y'] * 4)
        self.assertEqual(r.front, ['G'] * 2 + ['R'] * 2)
        self.assertEqual(r.left, ['O'] * 2 + ['G'] * 2)
        self.assertEqual(r.right, ['R'] * 2 + ['B'] * 2)

    def test_down(self):
        s = cube2x2.initial_state
        s = cube2x2.transform(s, cube2x2.Action.D)
        r = cube2x2.render(s)
        self.assertEqual(r.back, ['O'] * 2 + ['B'] * 2)
        self.assertEqual(r.bottom, ['Y'] * 4)
        self.assertEqual(r.front, ['R'] * 2 + ['G'] * 2)
        self.assertEqual(r.left, ['G'] * 2 + ['O'] * 2)
        self.assertEqual(r.right, ['B'] * 2 + ['R'] * 2)
        self.assertEqual(r.top, ['W'] * 4)

    def test_down_rev(self):
        s = cube2x2.initial_state
        s = cube2x2.transform(s, cube2x2.Action.d)
        r = cube2x2.render(s)
        self.assertEqual(r.back, ['O'] * 2 + ['G'] * 2)
        self.assertEqual(r.bottom, ['Y'] * 4)
        self.assertEqual(r.front, ['R'] * 2 + ['B'] * 2)
        self.assertEqual(r.left, ['G'] * 2 + ['R'] * 2)
        self.assertEqual(r.right, ['B'] * 2 + ['O'] * 2)
        self.assertEqual(r.top, ['W'] * 4)

    def test_right(self):
        s = cube2x2.initial_state
        s = cube2x2.transform(s, cube2x2.Action.R)
        r = cube2x2.render(s)
        self.assertEqual(r.back, ['W', 'O'] * 2)
        self.assertEqual(r.bottom, ['Y', 'O'] * 2)
        self.assertEqual(r.front, ['R', 'Y'] * 2)
        self.assertEqual(r.left, ['G'] * 4)
        self.assertEqual(r.right, ['B'] * 4)
        self.assertEqual(r.top, ['W', 'R'] * 2)

    def test_right_rev(self):
        s = cube2x2.initial_state
        s = cube2x2.transform(s, cube2x2.Action.r)
        r = cube2x2.render(s)
        self.assertEqual(r.back, ['Y', 'O'] * 2)
        self.assertEqual(r.bottom, ['Y', 'R'] * 2)
        self.assertEqual(r.front, ['R', 'W'] * 2)
        self.assertEqual(r.left, ['G'] * 4)
        self.assertEqual(r.right, ['B'] * 4)
        self.assertEqual(r.top, ['W', 'O'] * 2)

    def test_left(self):
        s = cube2x2.initial_state
        s = cube2x2.transform(s, cube2x2.Action.L)
        r = cube2x2.render(s)
        self.assertEqual(r.back, ['O', 'Y'] * 2)
        self.assertEqual(r.bottom, ['R', 'Y'] * 2)
        self.assertEqual(r.front, ['W', 'R'] * 2)
        self.assertEqual(r.left, ['G'] * 4)
        self.assertEqual(r.right, ['B'] * 4)
        self.assertEqual(r.top, ['O', 'W'] * 2)

    def test_left_rev(self):
        s = cube2x2.initial_state
        s = cube2x2.transform(s, cube2x2.Action.l)
        r = cube2x2.render(s)
        self.assertEqual(r.back, ['O', 'W'] * 2)
        self.assertEqual(r.bottom, ['O', 'Y'] * 2)
        self.assertEqual(r.front, ['Y', 'R'] * 2)
        self.assertEqual(r.left, ['G'] * 4)
        self.assertEqual(r.right, ['B'] * 4)
        self.assertEqual(r.top, ['R', 'W'] * 2)

    def test_front(self):
        s = cube2x2.initial_state
        s = cube2x2.transform(s, cube2x2.Action.F)
        r = cube2x2.render(s)
        self.assertEqual(r.back, ['O'] * 4)
        self.assertEqual(r.bottom, ['B'] * 2 + ['Y'] * 2)
        self.assertEqual(r.front, ['R'] * 4)
        self.assertEqual(r.left, ['G', 'Y'] * 2)
        self.assertEqual(r.right, ['W', 'B'] * 2)
        self.assertEqual(r.top, ['W'] * 2 + ['G'] * 2)

    def test_front_rev(self):
        s = cube2x2.initial_state
        s = cube2x2.transform(s, cube2x2.Action.f)
        r = cube2x2.render(s)
        self.assertEqual(r.back, ['O'] * 4)
        self.assertEqual(r.bottom, ['G'] * 2 + ['Y'] * 2)
        self.assertEqual(r.front, ['R'] * 4)
        self.assertEqual(r.left, ['G', 'W'] * 2)
        self.assertEqual(r.right, ['Y', 'B'] * 2)
        self.assertEqual(r.top, ['W'] * 2 + ['B'] * 2)

    def test_back(self):
        s = cube2x2.initial_state
        s = cube2x2.transform(s, cube2x2.Action.B)
        r = cube2x2.render(s)
        self.assertEqual(r.back, ['O'] * 4)
        self.assertEqual(r.bottom, ['Y'] * 2 + ['G'] * 2)
        self.assertEqual(r.front, ['R'] * 4)
        self.assertEqual(r.left, ['W', 'G'] * 2)
        self.assertEqual(r.right, ['B', 'Y'] * 2)
        self.assertEqual(r.top, ['B'] * 2 + ['W'] * 2)

    def test_back_rev(self):
        s = cube2x2.initial_state
        s = cube2x2.transform(s, cube2x2.Action.b)
        r = cube2x2.render(s)
        self.assertEqual(r.back, ['O'] * 4)
        self.assertEqual(r.bottom, ['Y'] * 2 + ['B'] * 2)
        self.assertEqual(r.front, ['R'] * 4)
        self.assertEqual(r.left, ['Y', 'G'] * 2)
        self.assertEqual(r.right, ['B', 'W'] * 2)
        self.assertEqual(r.top, ['G'] * 2 + ['W'] * 2)

    def test_inverse_right(self):
        s = cube2x2.initial_state
        s = cube2x2.transform(s, cube2x2.Action.R)
        s = cube2x2.transform(s, cube2x2.Action.r)
        self.assertEqual(s, cube2x2.initial_state)

        s = cube2x2.initial_state
        s = cube2x2.transform(s, cube2x2.Action.r)
        s = cube2x2.transform(s, cube2x2.Action.R)
        self.assertEqual(s, cube2x2.initial_state)

    def test_inverse_left(self):
        s = cube2x2.initial_state
        s = cube2x2.transform(s, cube2x2.Action.L)
        s = cube2x2.transform(s, cube2x2.Action.l)
        self.assertEqual(s, cube2x2.initial_state)

        s = cube2x2.initial_state
        s = cube2x2.transform(s, cube2x2.Action.l)
        s = cube2x2.transform(s, cube2x2.Action.L)
        self.assertEqual(s, cube2x2.initial_state)

    def test_inverse_top(self):
        s = cube2x2.initial_state
        s = cube2x2.transform(s, cube2x2.Action.T)
        s = cube2x2.transform(s, cube2x2.Action.t)
        self.assertEqual(s, cube2x2.initial_state)

        s = cube2x2.initial_state
        s = cube2x2.transform(s, cube2x2.Action.t)
        s = cube2x2.transform(s, cube2x2.Action.T)
        self.assertEqual(s, cube2x2.initial_state)

    def test_inverse_down(self):
        s = cube2x2.initial_state
        s = cube2x2.transform(s, cube2x2.Action.D)
        s = cube2x2.transform(s, cube2x2.Action.d)
        self.assertEqual(s, cube2x2.initial_state)

        s = cube2x2.initial_state
        s = cube2x2.transform(s, cube2x2.Action.d)
        s = cube2x2.transform(s, cube2x2.Action.D)
        self.assertEqual(s, cube2x2.initial_state)

    def test_inverse_front(self):
        s = cube2x2.initial_state
        s = cube2x2.transform(s, cube2x2.Action.F)
        s = cube2x2.transform(s, cube2x2.Action.f)
        self.assertEqual(s, cube2x2.initial_state)

        s = cube2x2.initial_state
        s = cube2x2.transform(s, cube2x2.Action.f)
        s = cube2x2.transform(s, cube2x2.Action.F)
        self.assertEqual(s, cube2x2.initial_state)

    def test_inverse_back(self):
        s = cube2x2.initial_state
        s = cube2x2.transform(s, cube2x2.Action.B)
        s = cube2x2.transform(s, cube2x2.Action.b)
        self.assertEqual(s, cube2x2.initial_state)

        s = cube2x2.initial_state
        s = cube2x2.transform(s, cube2x2.Action.b)
        s = cube2x2.transform(s, cube2x2.Action.B)
        self.assertEqual(s, cube2x2.initial_state)

    def test_inverse(self):
        s = cube2x2.initial_state
        for a in cube2x2.Action:
            s = cube2x2.transform(s, a)
            r = cube2x2.render(s)
            s = cube2x2.transform(s, cube2x2.inverse_action(a))
            r2 = cube2x2.render(s)
        self.assertEqual(s, cube2x2.initial_state)

    def test_sequence(self):
        acts = [cube2x2.Action.R, cube2x2.Action.t, cube2x2.Action.R, cube2x2.Action.D, cube2x2.Action.F,
                cube2x2.Action.d, cube2x2.Action.T, cube2x2.Action.R, cube2x2.Action.D, cube2x2.Action.F]

        s = cube2x2.initial_state
        for a in acts:
            s = cube2x2.transform(s, a)
        r = cube2x2.render(s)
        for a in reversed(acts):
            s = cube2x2.transform(s, cube2x2.inverse_action(a))
        r = cube2x2.render(s)
        self.assertEqual(s, cube2x2.initial_state)


class CubeEncoding(unittest.TestCase):
    def test_init(self):
        tgt = np.zeros(shape=cube2x2.encoded_shape)
        s = cube2x2.initial_state
        cube2x2.encode_inplace(tgt, s)

    def test_random(self):
        s = cube2x2.initial_state
        for _ in range(200):
            a = cube2x2.Action(random.randrange(len(cube2x2.Action)))
            s = cube2x2.transform(s, a)
            tgt = np.zeros(shape=cube2x2.encoded_shape)
            cube2x2.encode_inplace(tgt, s)
            self.assertEqual(tgt.sum(), 8)


if __name__ == '__main__':
    unittest.main()
