# Copyright 2019 The ROBEL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper methods and classes for plotting data."""

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.animation as animation


class AnimatedPlot:
    """Plots data that can be updated over time."""

    def __init__(self, bounds: Optional[Sequence[float]] = None):
        """Initializes a new plot."""
        self.fig, self.ax = plt.subplots()
        self.ani = None
        self.update_fn = None
        self.iteration = 0
        self.elements = []

        if bounds:
            self.ax.axis(bounds)
        self.ax.grid()

    @property
    def is_open(self) -> bool:
        """Returns True if the figure window is open."""
        return plt.fignum_exists(self.fig.number)

    def add(self, element):
        """Adds an element for animation."""
        self.elements.append(element)

    def show(self, frame_rate: int = 30, blocking: bool = False):
        """Displays the plot."""
        self.ani = animation.FuncAnimation(
            self.fig,
            self._update,
            interval=1000 // frame_rate,
            init_func=self._init,
            blit=False,
        )
        plt.show(block=blocking)

    def refresh(self):
        """Allows the GUI to update."""
        self.fig.canvas.draw()
        self.fig.canvas.start_event_loop(0.0001)

    def _init(self):
        return self.elements

    def _update(self, iteration):
        self.iteration = iteration
        if self.update_fn is not None:
            self.update_fn()  # pylint: disable=not-callable
        return self.elements
