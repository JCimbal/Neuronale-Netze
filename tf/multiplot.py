from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plt.subplots_adjust(right=0.75)

host = host_subplot(111, axes_class=AA.Axes)
host.set_xlim(0, 2)
host.set_ylim(0, 2)
host.set_xlabel("Distance")
host.set_ylabel("Density")

# densGraph = host.twinx()

tempGraph = host.twinx()
tempGraph.set_ylabel("Temperature")
tempGraph.set_ylim(0, 4)

velGraph = host.twinx()
velGraph.set_ylabel("Velocity")
velGraph.set_ylim(1, 65)

new_fixed_axis = velGraph.get_grid_helper().new_fixed_axis
velGraph.axis["right"] = new_fixed_axis(loc="right", axes=velGraph,offset=(60, 0))
velGraph.axis["right"].toggle(all=True)

p1, = host.plot([0, 1, 2], [0, 1, 2], label="Density")
p2, = tempGraph.plot([0, 1, 2], [0, 3, 2], label="Temperature")
p3, = velGraph.plot([0, 1, 2], [50, 30, 15], label="Velocity")

host.legend()

host.axis["left"].label.set_color(p1.get_color())
tempGraph.axis["right"].label.set_color(p2.get_color())
velGraph.axis["right"].label.set_color(p3.get_color())

plt.draw()
plt.show()
