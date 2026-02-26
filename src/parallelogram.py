import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import math

tolerance = 1e-6

### parallelogram wrappers

@dataclass
class Parallelogram:
    "Parallelogram with basis vectors v1 and v2, wrap on Z2"
    v1: np.array
    v2: np.array

    def __post_init__(self):
        self.v1 = np.array(self.v1)
        self.v2 = np.array(self.v2)
        assert len(self.v1) == 2
        assert len(self.v2) == 2

        self.V = np.vstack((self.v1,self.v2)).T
        self.all_points = self.get_all_points()

    
    def get_all_points(self) -> np.array:
        "Generate a list of points in parallelogram"

        minx = -2*abs(max(self.v1[0], self.v2[0]))
        maxx = 2*abs(max(self.v1[0], self.v2[0]))

        miny = -2*abs(max(self.v1[1], self.v2[1]))
        maxy = 2*abs(max(self.v1[1], self.v2[1]))

        #get all points
        points_in_para = []
        for xind in range(minx,maxx,1):
            for yind in range(miny,maxy,1):
                consid_point_ind = (np.linalg.inv(self.V) @ [xind, yind]).astype(np.float16)
                if consid_point_ind[0] >= 0 and consid_point_ind[0] < 1:
                    if consid_point_ind[1] >= 0 and consid_point_ind[1] < 1:
                        points_in_para.append((xind,yind))
        print(len(points_in_para), "points in parallelogram")     
        return points_in_para


    def wrap_point(self,newp: np.array) -> np.array:
        "Wrap around periodic boundary conditions"
        V = self.V
        basisrep = (np.linalg.inv(V) @ newp).astype(np.float16)
        pararep = [basisrep[0] - math.floor(basisrep[0]), basisrep[1] - math.floor(basisrep[1]) ]
        return tuple(map(int, tuple(np.rint(V @ pararep))))