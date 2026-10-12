import tempfile
import unittest
from pathlib import Path

import numpy as np

import aura_scene_capture as capture


class SceneCaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_root = capture.CAPTURE_ROOT
        capture.CAPTURE_ROOT = Path(self.tmp.name).resolve()
        capture.CAPTURE_ROOT.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        capture.CAPTURE_ROOT = self.old_root
        self.tmp.cleanup()

    def test_camera_upload_is_persisted(self):
        meta = capture.create_capture(b"fake-webm-bytes", content_type="video/webm", name="office", duration_s=23.4)
        self.assertEqual(meta["status"], "uploaded")
        self.assertEqual(meta["name"], "office")
        self.assertAlmostEqual(meta["duration_s"], 23.4)
        directory = capture.CAPTURE_ROOT / meta["id"]
        self.assertTrue((directory / meta["video_file"]).is_file())
        self.assertFalse(meta["splat_ready"])

    def test_binary_gaussian_ply_converts_to_web_splat(self):
        props = [
            "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
            "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        ]
        header = [
            "ply", "format binary_little_endian 1.0", "element vertex 2",
            *[f"property float {name}" for name in props], "end_header", "",
        ]
        rows = np.asarray([
            [0, 1, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0],
            [3, 4, 5, .1, -.1, .2, 1, np.log(2), np.log(3), np.log(4), 1, 0, 0, 0],
        ], dtype="<f4")
        root = capture.CAPTURE_ROOT / "ply-test"
        root.mkdir()
        ply = root / "scene.ply"
        out = root / "scene.splat"
        ply.write_bytes("\n".join(header).encode("ascii") + rows.tobytes())
        count = capture._convert_gaussian_ply_to_splat(ply, out)
        self.assertEqual(count, 2)
        self.assertEqual(out.stat().st_size, 64)
        raw = out.read_bytes()
        # Records are sorted most visible first (volume × opacity): row 2 (volume 24) leads.
        first = np.frombuffer(raw[:24], dtype="<f4")
        second = np.frombuffer(raw[32:56], dtype="<f4")
        np.testing.assert_allclose(first[:3], [3, 5, -4], atol=1e-6)
        np.testing.assert_allclose(first[3:6], [2, 3, 4], atol=1e-5)
        np.testing.assert_allclose(second[:3], [0, 2, -1], atol=1e-6)
        np.testing.assert_allclose(second[3:6], [1, 1, 1], atol=1e-6)
        # Identity orientation is rotated with the Z-up -> Y-up swap: q_x(-90°) = (√½, -√½, 0, 0).
        rot = np.frombuffer(raw[60:64], dtype=np.uint8).astype(float)
        np.testing.assert_allclose((rot - 128) / 128, [np.sqrt(.5), -np.sqrt(.5), 0, 0], atol=1 / 128)


if __name__ == "__main__":
    unittest.main()
