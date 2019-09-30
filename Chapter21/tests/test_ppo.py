import unittest
from numpy import testing

from lib import ppo


class TestPPO(unittest.TestCase):
    def test_adv_ref(self):
        vals = [0, 0, 0, 0, 0]
        dones = [False, False, True, False, False]
        rewards = [1, 1, 1, 1, 1]

        adv_t, ref_t = ppo.calc_adv_ref(vals, dones, rewards, 1.0, 1.0)
        adv = adv_t.detach().numpy()
        ref = ref_t.detach().numpy()

        testing.assert_array_equal(ref, [3, 2, 1, 1])
        testing.assert_array_equal(ref, adv)

        adv_t, ref_t = ppo.calc_adv_ref(vals, dones, rewards, 0.9, 1.0)
        adv = adv_t.detach().numpy()
        ref = ref_t.detach().numpy()

        testing.assert_array_almost_equal(ref, [2.71, 1.9, 1., 1.])
        testing.assert_array_almost_equal(ref, adv)


        pass

