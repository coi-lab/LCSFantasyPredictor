"""Unit tests for Riot Match-V5 API client and rate limiter."""

import unittest
from champion_prediction.riot_match_v5 import RiotMatchV5Client


class RiotMatchV5ClientTests(unittest.TestCase):

    def setUp(self) -> None:
        self.client = RiotMatchV5Client()

    def test_client_initialization(self) -> None:
        self.assertIsNotNone(self.client.api_key)

    def test_invalid_key_graceful_handling(self) -> None:
        client_bad_key = RiotMatchV5Client(api_key="INVALID_KEY")
        res = client_bad_key.fetch_challenger_league(region="kr")
        self.assertIn("error", res)


if __name__ == "__main__":
    unittest.main()
