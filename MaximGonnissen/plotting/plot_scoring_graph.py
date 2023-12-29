import matplotlib.pyplot as plt


class Result:
    """
    Class to store the results of a seasonality submission.
    """
    def __init__(self, private_score: float, public_score: float, day_offset: int, day_range: int = 30):
        self.private_score = private_score
        self.public_score = public_score
        self.day_offset = day_offset
        self.day_range = day_range


def generate_graph(scores: list, day_range: int = 30):
    """
    Generates a graph of the scores.
    :param scores: List of scores
    :param day_range: Day range to use for the graph title
    :return: None
    """
    fig, ax = plt.subplots()
    ax.plot([score.day_offset for score in scores], [score.private_score for score in scores], label='Private score')
    ax.plot([score.day_offset for score in scores], [score.public_score for score in scores], label='Public score')
    ax.set_xlabel('Day offset')
    ax.set_ylabel('Score')
    ax.set_title(f'Scores (day range: {day_range})')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    results_30 = [
        Result(0.00042, 0.00035, -30),
        Result(0.00063, 0.00066, -60),
        Result(0.00014, 0.00011, -90),
        Result(0.00161, 0.00169, -120),
        Result(0.00031, 0.00033, -125),
        Result(0.00027, 0.00033, -45),
        Result(0.0003, 0.00021, -75),
        Result(0.00036, 0.00051, -110),
        Result(0.00022, 0.00025, -130),
        Result(0.00087, 0.00074, -115),
        Result(0.00039, 0.00034, -65),
        Result(0.00113, 0.00123, -55),
        Result(0.00212, 0.00253, 0),
        Result(0.00178, 0.00225, -15),
        Result(0.00089, 0.00108, -20),
    ]

    results_14 = [
        Result(0.00058, 0.00074, -120, 14)
    ]

    results = sorted(results_30, key=lambda x: x.day_offset)

    generate_graph(results)
