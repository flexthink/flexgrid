import unittest

from flexgrid.space import linear


class LinearTest(unittest.TestCase):
    def test_increment_is_the_default_scale(self):
        self.assertEqual(linear(1, 20, 5), [1, 6, 11, 16, 21])

    def test_gauge_uses_step_multiples_and_includes_endpoints(self):
        self.assertEqual(
            linear(1, 20, 5, scale="gauge"),
            [1, 5, 10, 15, 20],
        )

    def test_gauge_includes_non_multiple_upper_bound(self):
        self.assertEqual(
            linear(1, 18, 5, scale="gauge"),
            [1, 5, 10, 15, 18],
        )

    def test_gauge_does_not_duplicate_multiple_endpoints(self):
        self.assertEqual(linear(5, 20, 5, scale="gauge"), [5, 10, 15, 20])


if __name__ == "__main__":
    unittest.main()
