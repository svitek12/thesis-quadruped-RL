import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

class LivePlot:
    def __init__(self, draw_every=3, figsize=(6, 4), window_position=(1720, 90)):
        plt.ion()

        self.fig, (self.ax1, self.ax2) = plt.subplots(nrows=2, sharex=True, figsize=figsize)
        self.line1, = self.ax1.plot([], [], 'b-', label="Cumulated reward")  # blue line
        self.line2, = self.ax2.plot([], [], 'r-', label="Robot X position")  # red line

        self.ax1.grid(True)
        self.ax2.grid(True)

        #set titles
        self.ax1.set_title("Cumulated reward")
        self.ax2.set_title("Robot X position")

        self.x_data = []
        self.y1_data = []
        self.y2_data = []

        self.draw_every = max(1, draw_every)
        self._updates_since_draw = 0

        self.fig.tight_layout()
        self.fig.show()
        self.fig.canvas.draw()

        self._position_window(*window_position)

    def _position_window(self, x, y):
        """Przesuwa okno wykresu na stałą pozycję (x, y) ekranu, nie zmieniając
        jego rozmiaru. Rozmiar samego wykresu pilnuje `figsize`/`dpi` figury -
        wymuszanie tu też rozmiaru *okna* (razem z toolbarem/ramką) robi canvas
        mniejszym niż figura i przycina wykresy, więc tego nie robimy.
        Działa dla backendów Qt/Tk/WX - dla innych (np. inline w Jupyterze,
        gdzie nie ma osobnego okna) po prostu nic nie robi."""
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is None or getattr(manager, "window", None) is None:
            return

        backend = matplotlib.get_backend().lower()
        window = manager.window

        try:
            if "qt" in backend:
                window.move(x, y)
            elif "tk" in backend:
                window.wm_geometry(f"+{x}+{y}")
            elif "wx" in backend:
                window.SetPosition((x, y))
        except Exception as e:
            print(f"[LivePlot] Nie udało się ustawić pozycji okna: {e}")

    def update(self, new_x, new_y1, new_y2):
        """Append new data and, every `draw_every` calls, redraw both plots."""
        self.x_data.append(new_x)
        self.y1_data.append(new_y1)
        self.y2_data.append(new_y2)

        self.line1.set_data(self.x_data, self.y1_data)
        self.line2.set_data(self.x_data, self.y2_data)

        self._updates_since_draw += 1
        if self._updates_since_draw < self.draw_every:
            return
        self._updates_since_draw = 0

        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def clear(self):
        """Clear all data and reset both plots."""
        self.x_data = []
        self.y1_data = []
        self.y2_data = []
        self._updates_since_draw = 0

        self.line1.set_data([], [])
        self.line2.set_data([], [])

        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

class LiveStepPlot:
    def __init__(self, num_plots=8):
        super().__init__()

        plt.ion()

        self.num_plots = num_plots
        self.rows = 2
        self.cols = self.num_plots // self.rows

        self.fig, self.axes = plt.subplots(nrows=self.rows, ncols=self.cols, figsize=(15, 6), sharex=True)
        self.lines = []
        self.x_data = [deque(maxlen=25) for _ in range(self.num_plots)]
        self.y_data = [deque(maxlen=25) for _ in range(self.num_plots)]
        
        motor_dict = {
            0: "Przednie lewe biodro",
            1: "Tylne lewe biodro",
            2: "Przednie lewe kolano",
            3: "Przednie prawe biodro",

        }
        for i in range(self.rows):
            for j in range(self.cols):
                idx = i * self.cols + j
                ax = self.axes[i][j]
                line, = ax.step([], [], where='post')
                self.lines.append(line)
                ax.grid(True)
                ax.legend(loc="upper left")
                ax.set_title(motor_dict[idx])

        self.fig.tight_layout()
        self.fig.show()
        self.fig.canvas.draw()

    def update(self, new_x, new_ys):
        if len(new_ys) != self.num_plots:
            raise ValueError(f"new_ys must have {self.num_plots} elements.")

        for i in range(self.num_plots):
            self.x_data[i].append(new_x)
            self.y_data[i].append(new_ys[i])
            self.lines[i].set_data(self.x_data[i], self.y_data[i])

            ax = self.axes[i // self.cols][i % self.cols]
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def clear(self):
        for i in range(self.num_plots):
            self.x_data[i].clear()
            self.y_data[i].clear()
            self.lines[i].set_data([], [])
            
            ax = self.axes[i // self.cols][i % self.cols]
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
