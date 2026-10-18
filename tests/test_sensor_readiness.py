"""Check sensor readiness without a running simulator (source Humble first)."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/drl_navigation_ros2"))
from ros_python import ROS_env


class SensorReadinessTests(unittest.TestCase):
    def make_env(self):
        env = ROS_env.__new__(ROS_env)
        env.cmd_vel_publisher = Mock()
        env.physics_client = Mock()
        sensor = SimpleNamespace(latest_scan=None, latest_position=None, latest_heading=None)
        sensor.get_latest_sensor = lambda: (sensor.latest_scan, sensor.latest_position, sensor.latest_heading)
        env.sensor_subscriber = sensor
        env.get_dist_sincos = Mock(return_value=(2.0, 1.0, 0.0, 0.0))
        env.check_collision = Mock(return_value=False)
        env.check_target = Mock(return_value=False)
        return env

    def test_waits_for_both_topics_and_discards_previous_step(self):
        env = self.make_env()
        sensor = env.sensor_subscriber
        sensor.latest_scan, sensor.latest_position, sensor.latest_heading = [9.0], object(), object()
        received_position, received_heading = object(), object()
        calls = []

        def spin(node, timeout_sec):
            calls.append(timeout_sec)
            if len(calls) == 1:
                node.latest_scan = [2.0] * 360
            else:
                node.latest_position, node.latest_heading = received_position, received_heading

        with patch("ros_python.time.sleep"), patch("ros_python.rclpy.ok", return_value=True), patch("ros_python.rclpy.spin_once", side_effect=spin):
            result = env.step()
        self.assertEqual(len(calls), 2)
        self.assertEqual(result[0], [2.0] * 360)
        env.get_dist_sincos.assert_called_once_with(received_position, received_heading)
        env.physics_client.pause_physics.assert_called_once()

    def test_timeout_pauses_physics(self):
        env = self.make_env()
        with patch("ros_python.time.sleep"), patch("ros_python.time.monotonic", side_effect=[0.0, 6.0]), patch("ros_python.rclpy.spin_once") as spin:
            with self.assertRaises(TimeoutError):
                env.step()
        spin.assert_not_called()
        env.physics_client.pause_physics.assert_called_once()
        env.get_dist_sincos.assert_not_called()


if __name__ == "__main__":
    unittest.main()
