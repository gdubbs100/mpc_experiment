import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
from typing import Callable

class MotionViewer:
    def __init__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        landscape_func: Callable[[float], float],
        target_x: float,
        fps: int = 30,
        dt: float = 0.05,
        xlim=None,
    ):
        self.positions = positions
        self.velocities = velocities
        self.landscape_func = landscape_func
        self.target_x = target_x
        self.fps = fps
        self.dt = dt

        # --- x limits ---
        self.xlim = xlim or (
            min(target_x, positions.min()) - 1.0,
            max(target_x, positions.max()) + 1.0,
        )

        # --- sample landscape over full domain ---
        xs = np.linspace(*self.xlim, 1000)
        ys = np.array([self.landscape_func(x) for x in xs])

        # --- y limits with padding ---
        y_pad = 0.1 * (ys.max() - ys.min() + 1e-6)
        self.ylim = (ys.min() - y_pad, ys.max() + y_pad)

        # --- figure ---
        self.fig, self.ax = plt.subplots()
        self.dot, = self.ax.plot([], [], "ko", markersize=8)
        self.ax.axvline(target_x, color="red", linewidth=2)

        # --- landscape ---
        self.ax.plot(xs, ys, color="grey", linewidth=2, alpha=0.8)

        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.ax.set_yticks([])
        self.ax.set_xlabel("x")

        # --- motion ---
        self.frames = self._interpolate_motion()

    def render_frames(self):
        for i in range(len(self.frames)):
            self._update(i)
            self.fig.canvas.draw()

            # Get RGBA buffer from canvas
            buf = np.asarray(self.fig.canvas.buffer_rgba())
            # Drop alpha channel
            img = buf[:, :, :3]

            yield img


    def _interpolate_motion(self):
        frames = []
        substeps = max(1, int(self.fps * self.dt))

        for x0, v in zip(self.positions, self.velocities):
            for i in range(substeps):
                t = (i / substeps) * self.dt
                frames.append(x0 + v * t)

        return np.array(frames)

    def _update(self, i):
        x = self.frames[i]
        y = self.landscape_func(x)

        self.dot.set_data([x], [y])

        return (self.dot,)

    def save(self, filename="motion.mp4"):
        
        with imageio.get_writer(filename, fps=self.fps) as writer:
            for frame in self.render_frames():
                writer.append_data(frame)


