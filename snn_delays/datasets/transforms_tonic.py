import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class OnePolariy:
    """
    Select only the zero polarity. This transform does not have any
    parameters.

    Example:
        transform = tonic.transforms.MergePolarities()
    """
    def __call__(self, events):
        events = events.copy()
        filtered_events = []
        for event in events:
            if event[2] == False: # if polarity == 0
                filtered_events.append(event)
        return np.array(filtered_events)


class MergePolariy:
    """
    Select only the zero polarity. This transform does not have any
    parameters.

    Example:
        transform = tonic.transforms.MergePolarities()
    """
    def __call__(self, events):
        events = events.copy()
        events["p"] = np.zeros_like(events["p"])
        return events


@dataclass
class CropTimeRandom:
    """
    Custom CropTime Class
    Random crops
    """

    # min: int = None
    max_start: int = None
    duration: int = None

    def __call__(self, events):

        #self.max = 1e6
        #self.max = np.random.randint(200000, 1.5e6)
        # self.max = np.random.choice([100000, 1e6])

        #start =  int(1e4*np.random.rand())
        start = np.random.randint(0, self.max_start)
        end = start + self.duration

        print(start)

        assert "t" in events.dtype.names
        # if self.max is None:
        #     self.max = np.max(events["t"])
        #return events[(events["t"] >= self.min) & (events["t"] <= self.max)]
        #return events[(events["t"] >= start) & (events["t"] <= start+duration)]
        return events[(events["t"] >= start) & (events["t"] <= end)]