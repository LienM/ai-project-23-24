from typing import Collection
import time
from math import floor


class ASCIIColour:
    """
    Static class containing ASCII colour codes.
    """
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'


class ProgressBar:
    """
    Simple progress bar to display for long-running iterations.
    """

    def __init__(self, collection: Collection, bar_length: int = 20, show_count: bool = True,
                 clear_when_done: bool = True, show_estimate: bool = True, full_bar: str = '█', half_bar: str = '▌',
                 empty_bar: str = '░', bar_start: str = '[', bar_end: str = ']', ansi_colour: str = None,
                 parent_progress_bar: 'ProgressBar' = None):
        """
        Create a new progress bar.
        :param bar_length: Length of the progress bar.
        :param show_count: Whether to show the current count.
        :param clear_when_done: Whether to clear the progress bar when done.
        :param show_estimate: Whether to show the estimated time remaining.
        :param full_bar: Character to use for a full bar.
        :param half_bar: Character to use for a half bar.
        :param empty_bar: Character to use for an empty bar.
        :param bar_start: Character to use for the start of the bar.
        :param bar_end: Character to use for the end of the bar.
        :param ansi_colour: Colour to use for the progress bar.
        :param parent_progress_bar: Parent progress bar to nest this progress bar in.
        """
        self.collection = collection
        self.bar_length = bar_length
        self.show_count = show_count
        self.clear_when_done = clear_when_done
        self.show_estimate = show_estimate
        self.full_bar = full_bar
        self.half_bar = half_bar
        self.empty_bar = empty_bar
        self.bar_start = bar_start
        self.bar_end = bar_end
        self.progress_bar_colour = ansi_colour
        self.parent_progress_bar = parent_progress_bar

        self.children_progress_bars = []

        if self.parent_progress_bar is not None:
            self.parent_progress_bar.children_progress_bars.append(self)

        self.count = 0
        self.total = len(collection)
        self.start_time = time.time()

    def has_parent(self):
        return self.parent_progress_bar is not None

    def nested_padding(self):
        if self.has_parent():
            return self.parent_progress_bar.get_update_string() + self.parent_progress_bar.nested_padding() + '\t'
        else:
            return ''

    def get_progress(self) -> float:
        """
        Get the progress.
        :return: Progress.
        """
        return self.count / self.total

    def get_estimated_time_remaining(self) -> float:
        """
        Get the estimated time remaining.
        :return: Estimated time remaining.
        """
        if self.count == 0:
            return 0
        return ((time.time() - self.start_time) / self.count) * (self.total - self.count)

    def __enter__(self):
        self.count = 0
        self.total = len(self.collection)
        self.start_time = time.time()
        self.update()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.clear_when_done:
            print('\r', end='')
        else:
            self.update()

    def get_update_string(self) -> str:
        update_string = f'\r{self.nested_padding()}'

        update_string += f'{self.progress_bar_colour or ""}{self.bar_start}'

        progress = self.count / self.total
        progress_bar_length = floor(progress * self.bar_length)

        update_string += self.full_bar * progress_bar_length

        if (progress * self.bar_length - progress_bar_length) >= 0.5:
            update_string += self.half_bar
            update_string += self.empty_bar * (self.bar_length - progress_bar_length - 1)
        else:
            update_string += self.empty_bar * (self.bar_length - progress_bar_length)

        update_string += self.bar_end

        if self.show_count:
            update_string += f' {self.count}/{self.total}'

        update_string += f' ({progress * 100:.2f}%)'

        update_string += f' {time.time() - self.start_time:.2f}s'

        if self.show_estimate:
            update_string += f' T-{self.get_estimated_time_remaining():.2f}s'

        update_string += ASCIIColour.RESET if self.progress_bar_colour is not None else ''

        return update_string

    def update(self):
        """
        Update the progress bar.
        """
        print(self.get_update_string() + '    ', end='')

    def iterate(self):
        """
        Iterate over the collection, updating the progress bar.
        """
        self.count = 0
        for _ in self.collection:
            self.count += 1
            self.update()
            yield _

    def __iter__(self):
        return self.iterate()


if __name__ == '__main__':
    class ExampleCollection:
        def __init__(self, length: int):
            self.length = length

        def __len__(self):
            return self.length

        def __iter__(self):
            for i in range(self.length):
                yield i


    with ProgressBar(ExampleCollection(100)) as pb:
        for i in pb:
            with ProgressBar(ExampleCollection(30000), parent_progress_bar=pb, show_estimate=False, show_count=False) as pb2:
                for j in pb2:
                    pass
