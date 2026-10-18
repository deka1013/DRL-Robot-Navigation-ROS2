"""Regression coverage for SAC log density at saturated action bounds."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/drl_navigation_ros2"))
from SAC.SAC_actor import SquashedNormal
from SAC.SAC import SAC


class SACNumericsTests(unittest.TestCase):
    def test_saturated_actions_have_finite_density_and_gradients(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                mean = torch.tensor([[100.0, -100.0]], device=device, requires_grad=True)
                log_std = torch.zeros_like(mean, requires_grad=True)
                action, log_prob = SquashedNormal(mean, log_std.exp()).rsample_with_log_prob()
                self.assertTrue(torch.equal(action.abs(), torch.ones_like(action)))
                self.assertTrue(torch.isfinite(log_prob).all())
                log_prob.sum().backward()
                self.assertTrue(torch.isfinite(mean.grad).all())
                self.assertTrue(torch.isfinite(log_std.grad).all())

    def test_update_frequencies_continue_across_training_calls(self):
        model = SAC.__new__(SAC)
        model.device = torch.device("cpu")
        model.step = model.gradient_step = 0
        model.save_every = 0
        model.writer = Mock()
        model.train_metrics_dict = {"train/batch_reward_av": []}
        model.actor_update_frequency = 2
        model.critic_target_update_frequency = 3
        model.critic_tau = 0.005
        model.critic, model.critic_target = Mock(), Mock()
        model.update_critic = Mock()
        model.update_actor_and_alpha = Mock()
        buffer = Mock()
        buffer.sample_batch.return_value = tuple(np.zeros((2, 1)) for _ in range(5))
        with patch("SAC.SAC.utils.soft_update_params") as target_update:
            model.train(buffer, iterations=4, batch_size=2)
            model.train(buffer, iterations=4, batch_size=2)
        self.assertEqual([call.args[1] for call in model.update_actor_and_alpha.call_args_list], [0, 2, 4, 6])
        self.assertEqual(target_update.call_count, 3)

    def test_density_matches_transform_away_from_bounds(self):
        dist = SquashedNormal(torch.zeros(40, 2), torch.full((40, 2), 0.2))
        action, log_prob = dist.rsample_with_log_prob()
        torch.testing.assert_close(log_prob, dist.log_prob(action).sum(-1, keepdim=True))


if __name__ == "__main__":
    unittest.main()
