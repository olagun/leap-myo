import matplotlib.pyplot as plt
from utils import Rx, Ry, Rz
import numpy as np
import mpl_toolkits.mplot3d as plt3d
import math

fig = plt.figure()

ax = fig.add_subplot(
    121, projection="3d", xlim=(300, -300), ylim=(300, -300), zlim=(-300, 300)
)

ax.view_init(elev=45, azim=122)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

x = []
y = []
z = []

basis = np.array([[0], [200], [0]])
vector = Rz(math.pi / 3).dot(basis)

x.append(vector[0, 0])
y.append(vector[1, 0])
z.append(vector[2, 0])

line = plt3d.art3d.Line3D([0, vector[0, 0]], [0, vector[1, 0]], [0, vector[2, 0]])
ax.add_line(line)

ax.scatter(x, y, z, s=[10] * len(x), alpha=1)

plt.show()
