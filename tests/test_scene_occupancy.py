import copy
import unittest

from aura_scene_lab import DEFAULT_SCENE, normalize_scene, plan_path, _segment_clear


class OccupancyPlanningTests(unittest.TestCase):
    def scene(self):
        scene = copy.deepcopy(DEFAULT_SCENE)
        scene['obstacles'] = []
        scene['occupancy'] = {'bounds': scene['bounds'], 'cell_size': 0.1,
                              'occupied': [[30, z] for z in range(15, 33)]}
        return scene

    def test_scanned_obstacle_changes_route(self):
        scene = self.scene()
        plan = plan_path(scene)
        self.assertGreater(len(plan['path']), 2)
        normalized = normalize_scene(scene)
        for a, b in zip(plan['path'], plan['path'][1:]):
            self.assertTrue(_segment_clear(a, b, normalized))

    def test_wall_cannot_be_crossed(self):
        scene = self.scene()
        scene['occupancy']['occupied'] = [[30, z] for z in range(48)]
        with self.assertRaisesRegex(ValueError, 'No collision-clear'):
            plan_path(scene)

    def test_exact_endpoint_checked(self):
        scene = self.scene()
        scene['start'] = [0.01, 0.01]
        with self.assertRaisesRegex(ValueError, 'start'):
            plan_path(scene)

    def test_room_clearance(self):
        scene = self.scene()
        scene['start'] = [-2.99, 0]
        with self.assertRaisesRegex(ValueError, 'start'):
            plan_path(scene)

    def test_exact_thin_obstacle_intersection(self):
        scene = normalize_scene(self.scene())
        scene['occupancy'] = None
        scene['obstacles'] = [{'x': 0.123, 'z': 0, 'width': 0.001, 'depth': 1}]
        self.assertFalse(_segment_clear((-1, 0), (1, 0), scene))

    def test_malformed_cell_rejected(self):
        scene = self.scene()
        scene['occupancy']['occupied'] = [[-1, 0]]
        with self.assertRaisesRegex(ValueError, 'cell index'):
            plan_path(scene)

    def test_same_grid_cell_preserves_endpoints(self):
        scene = self.scene()
        scene['start'], scene['goal'] = [-2, -1], [-1.99, -1]
        route = plan_path(scene)['path']
        self.assertEqual(route[0], scene['start'])
        self.assertEqual(route[-1], scene['goal'])


if __name__ == '__main__':
    unittest.main()
