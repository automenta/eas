import unittest
import torch
import numpy as np
from eas.src.watcher import EmergentWatcher

class TestEmergentWatcher(unittest.TestCase):
    def setUp(self):
        self.dim = 128
        self.k = 5
        self.watcher = EmergentWatcher(dim=self.dim, k=self.k)

    def test_initialization(self):
        self.assertEqual(self.watcher.dim, self.dim)
        self.assertEqual(self.watcher.k, self.k)
        self.assertEqual(self.watcher.attractor_memory.attractors.shape, (self.k, self.dim))

    def test_whitening_update(self):
        x = torch.randn(10, self.dim)
        self.watcher.whitening_buffer.update(x)
        self.assertTrue(self.watcher.whitening_buffer.initialized)

    def test_snap_shape(self):
        batch_size = 4
        seq_len = 10
        hidden = torch.randn(batch_size, seq_len, self.dim)

        # Must run update at least once to init whitening
        self.watcher.whitening_buffer.update(hidden.mean(dim=1))

        snapped = self.watcher.snap(hidden)
        self.assertEqual(snapped.shape, hidden.shape)

    def test_update_logic(self):
        # Simulate successful activations
        hidden = torch.randn(2, 10, self.dim)
        initial_attractors = self.watcher.attractor_memory.attractors.clone()

        # Force update count to be low to trigger immediate clustering if logic allows
        # Actually update frequency is 5 by default.
        for _ in range(10):
            self.watcher.update(hidden)

        # Check if update_count increased (meaning clustering ran)
        self.assertTrue(self.watcher.update_count >= 1)

if __name__ == '__main__':
    unittest.main()
