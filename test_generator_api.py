import json
import tempfile
from pathlib import Path

import numpy as np

from aura_text2motion_generator import AuraText2MotionGenerator, list_examples


def main():
    examples = list_examples()
    assert len(examples) >= 1
    assert all(x.get('prompt') for x in examples)
    with tempfile.TemporaryDirectory() as td:
        g = AuraText2MotionGenerator(Path(td))
        g._resolved_model = 'text2motion-aura-g1'
        frames = 10
        posed = np.zeros((1, frames, 34, 3), dtype=np.float32)
        posed[0, :, 0, 2] = np.linspace(0.0, 0.7, frames)
        rotations = np.tile(np.eye(3, dtype=np.float32), (1, frames, 34, 1, 1))
        item = g._publish({
            'posed_joints': posed,
            'global_rot_mats': rotations,
            'local_rot_mats': rotations.copy(),
        }, 30.0, 'generator test', batch_id='batch-test', candidate_index=1, candidate_count=2, generation_seed=42)
        assert item['frames'] == frames
        assert item['batch_id'] == 'batch-test' and item['candidate_index'] == 1 and item['candidate_count'] == 2
        native = Path(td) / item['native_file']
        preview = Path(td) / item['preview_file']
        assert native.is_file() and preview.is_file()
        payload = json.loads(preview.read_text())
        assert payload['format'] == 'g1-joints-v2'
        assert len(payload['positions']) == frames
    print('generator API helper tests passed', len(examples), 'Text2Motion Aura examples')


if __name__ == '__main__':
    main()
