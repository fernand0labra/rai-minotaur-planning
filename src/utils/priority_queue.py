import heapq
import itertools


class PriorityQueue:
    """
    Min-priority queue. The itertools counter breaks ties deterministically
    (otherwise heapq falls back to comparing the stored items, which
    breaks for tuples of mixed types like grid coords vs. hybrid states).
    """

    def __init__(self):
        self._heap = []
        self._counter = itertools.count()

    def push(self, priority, item):
        heapq.heappush(self._heap, (priority, next(self._counter), item))

    def pop(self):
        priority, _, item = heapq.heappop(self._heap)
        return item

    def empty(self):
        return len(self._heap) == 0

    def __len__(self):
        return len(self._heap)