from matplotlib import pyplot as plt
import numpy as np
from myo_api import Myo, emg_mode
import time
from collections import Counter
from multiprocessing import Queue, Process, Manager


class MultichannelPlot(object):
    def __init__(self, nchan=8, xlen=512):
        self.nchan = nchan
        self.xlen = xlen
        self.fig = plt.figure(figsize=(10, 8))
        self.axes = [
            self.fig.add_subplot(int(str(self.nchan) + "1" + str(i + 1)))
            for i in range(self.nchan)
        ]
        for i, ax in enumerate(self.axes):
            plt.sca(ax)
            plt.ylabel("Ch.%d" % (i + 1))
        self.set_ylim([-128, 128])
        self.graphs = [
            ax.plot(np.arange(self.xlen), np.zeros(self.xlen))[0] for ax in self.axes
        ]

    def set_ylim(self, lims):
        [(ax.set_ylim(lims)) for ax in self.axes]

    def update_plot(self, sig):
        for g, data in zip(self.graphs, sig):
            if len(data) < self.xlen:
                data = np.concatenate([np.zeros(self.xlen - len(data)), data])
            if len(data) > self.xlen:
                data = data[-self.xlen :]
            g.set_ydata(data)
        plt.draw()
        plt.pause(0.04)


def update_plot(emg_data):
    plotter = MultichannelPlot()

    while True:
        plotter.update_plot(np.array(emg_data).T)


def main():
    with Manager() as manager:
        m = Myo(None, mode=emg_mode.RAW)

        emg_data = manager.list()
        plotter_process = Process(target=update_plot, args=(emg_data,))

        counter = Counter()
        state = {}
        state["start_time"] = time.time()

        def proc_emg(emg, moving):
            emg_data.append(emg)

            state["curr_time"] = time.time()
            counter["samples"] += 1

            if state["curr_time"] - state["start_time"] > 1:
                counter["samples_per_sec"] = counter["samples"]
                counter["samples"] = 0

                sps = counter["samples_per_sec"]
                print(f"samples per second {sps}")

                state["start_time"] = time.time()

        m.connect()
        m.add_emg_handler(proc_emg)

        running = True

        plotter_process.start()

        while running:
            m.run()


if __name__ == "__main__":
    main()
