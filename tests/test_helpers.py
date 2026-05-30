"""
Unit tests for the helper utilities in python/utils/helpers.py.
"""

from __future__ import annotations

import datetime
import logging
import os
import shutil
import tempfile
import unittest
from python.utils.helpers import (
    setup_logging,
    ensure_dir,
    load_config,
    timestamp_to_microseconds,
    microseconds_to_timestamp,
    format_indian_number,
)


class TestHelpers(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_setup_logging(self) -> None:
        logger_name = "TestLogger"
        logger = setup_logging(logger_name)
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, logger_name)
        self.assertTrue(len(logger.handlers) > 0)

        # Call again to ensure handler is not duplicated
        initial_handlers_count = len(logger.handlers)
        logger_recalled = setup_logging(logger_name)
        self.assertEqual(len(logger_recalled.handlers), initial_handlers_count)

    def test_ensure_dir(self) -> None:
        test_path = os.path.join(self.temp_dir, "new_sub_dir")
        self.assertFalse(os.path.exists(test_path))
        ensure_dir(test_path)
        self.assertTrue(os.path.exists(test_path))
        self.assertTrue(os.path.isdir(test_path))

    def test_load_config(self) -> None:
        # Non-existent file should return empty dict
        config = load_config("non_existent_file.yaml")
        self.assertEqual(config, {})

        # Existing config
        config_path = os.path.join(self.temp_dir, "test_config.yaml")
        with open(config_path, "w") as f:
            f.write("key1: value1\nkey2: 42\n")

        loaded = load_config(config_path)
        self.assertEqual(loaded, {"key1": "value1", "key2": 42})

    def test_timestamp_conversions(self) -> None:
        dt = datetime.datetime(2024, 6, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
        us = timestamp_to_microseconds(dt)
        self.assertEqual(us, 1717243200000000)

        dt_back = microseconds_to_timestamp(us)
        self.assertEqual(dt_back, dt)

    def test_format_indian_number(self) -> None:
        self.assertEqual(format_indian_number(100), "100")
        self.assertEqual(format_indian_number(1000), "1,000")
        self.assertEqual(format_indian_number(10000), "10,000")
        self.assertEqual(format_indian_number(100000), "1,00,000")
        self.assertEqual(format_indian_number(1000000), "10,00,000")
        self.assertEqual(format_indian_number(10000000), "1,00,00,000")
        self.assertEqual(format_indian_number(2250000), "22,50,000")


if __name__ == "__main__":
    unittest.main()
