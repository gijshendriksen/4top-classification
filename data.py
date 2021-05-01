import dataclasses
from pprint import pformat
from typing import List


@dataclasses.dataclass()
class Object:
    """
    Representation of a single particle in a captured event.
    """

    name: str
    charge: int
    e: float
    pt: float
    eta: float
    phi: float

    @staticmethod
    def read_object(s: str) -> 'Object':
        """
        Reads a single object from a comma-separated string.
        """
        name, e, pt, eta, phi = s.split(',')

        if len(name) < 2:
            charge = 0
        elif name[1] == '-':
            charge = -1
        else:
            charge = 1

        return Object(
            name[0],
            charge,
            float(e),
            float(pt),
            float(eta),
            float(phi),
        )

    def __str__(self):
        return pformat(self.__dict__)

    def __lt__(self, other):
        return dataclasses.astuple(self) < dataclasses.astuple(other)

    def __le__(self, other):
        return dataclasses.astuple(self) <= dataclasses.astuple(other)

    def __gt__(self, other):
        return dataclasses.astuple(self) > dataclasses.astuple(other)

    def __ge__(self, other):
        return dataclasses.astuple(self) >= dataclasses.astuple(other)


@dataclasses.dataclass()
class Event:
    """
    Representation of a single event (i.e. a single sample) of the dataset.
    """

    eid: int
    pid: str
    weight: float
    met: float
    metphi: float
    objects: List[Object]

    @staticmethod
    def read_event(line: str) -> 'Event':
        """
        Reads the event from a line, where values and objects are separated by semicolons.
        """
        parts = line.rstrip('\n;').split(';')

        eid, pid, weight, met, metphi = parts[:5]
        objects = [Object.read_object(o) for o in parts[5:]]

        return Event(
            int(eid),
            pid,
            float(weight),
            float(met),
            float(metphi),
            objects,
        )

    def __str__(self):
        return pformat(self.__dict__)
