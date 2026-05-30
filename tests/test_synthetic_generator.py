"""
Unit tests for the synthetic data generator in python/data_generation/synthetic_generator.py.
"""

from __future__ import annotations

import datetime
import os
import shutil
import tempfile
import unittest
import pandas as pd
from python.data_generation.synthetic_generator import SyntheticDataGenerator


class TestSyntheticDataGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        temp_dir_posix = self.temp_dir.replace('\\', '/')
        # Create a mock config file
        self.config_path = os.path.join(self.temp_dir, "test_settings.yaml")
        with open(self.config_path, "w") as f:
            f.write(f"""
simulation:
  num_days: 1
  snapshot_interval_ms: 1000
  num_order_book_levels: 3
paths:
  data_raw: "{temp_dir_posix}"
""")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_generator_initialization(self) -> None:
        gen = SyntheticDataGenerator(config_path=self.config_path)
        self.assertEqual(gen.num_days, 1)
        self.assertEqual(gen.snapshot_interval_ms, 1000)
        self.assertEqual(gen.num_levels, 3)
        self.assertEqual(gen.output_dir, self.temp_dir.replace('\\', '/'))

    def test_generate_single_day(self) -> None:
        gen = SyntheticDataGenerator(config_path=self.config_path)
        # Use a short simulation by mocking/hacking elapsed event times if needed,
        # but let's test with a seed that works.
        # Since generating a full day (22500 seconds) takes a few seconds, let's verify
        # that it returns the expected dataframes.
        # Wait, does generating a single day take too long? In python, generating a single day for a symbol
        # takes around 1-3 seconds. Let's run it.
        symbol = "RELIANCE"
        date = datetime.date(2025, 3, 3)
        seed = 1001

        # We can temporarily patch the HawkesProcess or simulate function to return fewer events if we want,
        # but 1-3 seconds is very acceptable for tests. Let's run a test call.
        snap_df, trade_df, event_df = gen.generate_single_day(symbol, date, seed)

        self.assertIsInstance(snap_df, pd.DataFrame)
        self.assertIsInstance(trade_df, pd.DataFrame)
        self.assertIsInstance(event_df, pd.DataFrame)

        # Verify columns exist
        self.assertIn("timestamp", snap_df.columns)
        self.assertIn("symbol", snap_df.columns)
        self.assertIn("mid_price", snap_df.columns)
        self.assertIn("bid_price_1", snap_df.columns)
        self.assertIn("ask_price_1", snap_df.columns)

        self.assertIn("timestamp", trade_df.columns)
        self.assertIn("symbol", trade_df.columns)
        self.assertIn("price", trade_df.columns)
        self.assertIn("size", trade_df.columns)

        self.assertIn("timestamp", event_df.columns)
        self.assertIn("symbol", event_df.columns)
        self.assertIn("event_type", event_df.columns)


if __name__ == "__main__":
    unittest.main()
