import pandas as pd


class CalendarDay:
    min_day = 1
    max_day = 365

    def __init__(self, doy: int):
        """
        :param doy: Day of year
        """
        self.doy = doy
        self.clamp()

    def clamp(self):
        """
        Clamps the day of year between min and max day, with overflow and underflow correction.
        """
        self.doy = self.doy % self.max_day
        if self.doy < self.min_day:
            self.doy = -self.doy % self.max_day + self.max_day

    def __str__(self):
        return str(self.doy)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, CalendarDay):
            return self.doy == other.doy
        elif isinstance(other, int):
            return self.doy == other
        else:
            return False

    def __lt__(self, other):
        if isinstance(other, CalendarDay):
            return self.doy < other.doy
        elif isinstance(other, int):
            return self.doy < other
        else:
            raise TypeError(f'Cannot compare CalendarDay to {type(other)}.')

    def __le__(self, other):
        if isinstance(other, CalendarDay):
            return self.doy <= other.doy
        elif isinstance(other, int):
            return self.doy <= other
        else:
            raise TypeError(f'Cannot compare CalendarDay to {type(other)}.')

    def __gt__(self, other):
        if isinstance(other, CalendarDay):
            return self.doy > other.doy
        elif isinstance(other, int):
            return self.doy > other
        else:
            raise TypeError(f'Cannot compare CalendarDay to {type(other)}.')

    def __ge__(self, other):
        if isinstance(other, CalendarDay):
            return self.doy >= other.doy
        elif isinstance(other, int):
            return self.doy >= other
        else:
            raise TypeError(f'Cannot compare CalendarDay to {type(other)}.')

    def __hash__(self):
        return hash(self.doy)

    def __add__(self, other):
        if isinstance(other, CalendarDay):
            return CalendarDay(self.doy + other.doy)
        elif isinstance(other, int):
            return CalendarDay(self.doy + other)
        else:
            raise TypeError(f'Cannot add CalendarDay to {type(other)}.')

    def __sub__(self, other):
        if isinstance(other, CalendarDay):
            return CalendarDay(self.doy - other.doy)
        elif isinstance(other, int):
            return CalendarDay(self.doy - other)
        else:
            raise TypeError(f'Cannot subtract CalendarDay from {type(other)}.')

    def days_until(self, other) -> int:
        if not isinstance(other, CalendarDay):
            raise TypeError(f'Cannot calculate days between CalendarDay and {type(other)}.')

        if self.doy == other.doy:
            return 0
        elif self.doy > other.doy:
            return (self.max_day - self.doy) + other.doy
        else:
            return other.doy - self.doy

    def distance_from(self, other) -> int:
        if not isinstance(other, CalendarDay):
            raise TypeError(f'Cannot calculate days between CalendarDay and {type(other)}.')

        return min(self.days_until(other), other.days_until(self))


class Season:
    """
    Season class that holds information about a season.

    :cvar max_score_offset: Offset from start of season to max score day.
    :cvar max_score_day_range: Range of days around max score day that can be scored.
    """
    max_score_offset = -30
    max_score_day_range = 30

    def __init__(self, season_name: str, start_doy: CalendarDay, end_doy: CalendarDay):
        """
        :param season_name: Season name
        :param start_doy: Start day of year
        :param end_doy: End day of year
        """
        self.season_name = season_name
        if isinstance(start_doy, int):
            start_doy = CalendarDay(start_doy)
        self.start_doy = start_doy
        if isinstance(end_doy, int):
            end_doy = CalendarDay(end_doy)
        self.end_doy = end_doy
        self.max_score_day = self.start_doy + self.max_score_offset

    def __str__(self):
        return self.season_name

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, Season):
            return self.season_name == other.season_name
        else:
            return False

    def __lt__(self, other):
        if isinstance(other, Season):
            return self.start_doy < other.start_doy

    def __hash__(self):
        return hash(self.season_name)

    def season_length(self) -> int:
        """
        Returns the length of the season in days.
        :return: Length of the season in days.
        """
        return self.start_doy.distance_from(self.end_doy)

    def in_season(self, date: pd.Timestamp) -> bool:
        """
        Checks if a date is in the season.
        :param date: Date to check
        :return: True if date is in season, False otherwise.
        """
        date_doy = date.dayofyear
        calendar_day = CalendarDay(date_doy)
        return calendar_day.days_until(self.end_doy) <= self.season_length()

    def get_season_score(self, date: pd.Timestamp) -> float:
        """
        Returns the season score of the item for the given date.
        :param date: Date to get season score for.
        :return: Season score for given date.
        """
        date_doy = date.dayofyear
        calendar_day = CalendarDay(date_doy)
        distance_from = calendar_day.distance_from(self.max_score_day)
        return max(0, self.max_score_day_range - distance_from)


seasons = [
    Season('spring', 60, 151),
    Season('summer', 152, 243),
    Season('fall', 244, 334),
    Season('winter', 335, 59)
]
