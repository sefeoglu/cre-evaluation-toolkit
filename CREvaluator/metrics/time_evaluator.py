from time import perf_counter


def time_evaluator(start_time: float, end_time: float | None = None) -> float:
	"""Return elapsed seconds from a ``perf_counter`` start value."""
	end = perf_counter() if end_time is None else end_time
	elapsed = end - start_time
	if elapsed < 0:
		raise ValueError("end_time must not precede start_time")
	return elapsed