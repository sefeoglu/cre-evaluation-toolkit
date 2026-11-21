from collections.abc import Iterable

from ._metric_utils import mean


def reward(rewards: Iterable[float]) -> float:
	"""Return the arithmetic mean of observed rewards."""
	return mean(rewards, "rewards")