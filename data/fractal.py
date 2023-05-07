import copy


class Fractals:

    def __init__(self):
        self.candles = []
        # List of candles, candle is a tuple of (open_, high, low, close, time)
        # open_, high, low, close are floats and time is a string
        self.fractals = []
        # List of fractals, fractal is a tuple of (value, direction, time)
        # value is a float, direction can be one of "up" or "down" and time is a string
        self.last_fractal_idx = None
        # The index of the candle corresponding to the last found fractal

    def get_candle(self, candle):
        """
        Method to get candles. This has to be from the API
        :param candle: A tuple of (open_, high, low, close, time)
        :return: None
        """
        # Get candle
        self.candles.append(candle)
        if self.last_fractal_idx is not None:
            self.last_fractal_idx -= 1
        # Check for fractals only if there are at least 5 candles
        if len(self.candles) > 4:
            if self.last_fractal_idx is not None:
                candle_idx = self.last_fractal_idx
            else:
                candle_idx = (len(self.candles) - 1) * -1
            # Check for fractals on all candles since candle_idx
            for idx in range(candle_idx+1, -2, 1):
                fractal = self.find_fractal(idx)
                if fractal is not None:
                    for f in fractal:
                        self.fractals.append(f)
                    self.last_fractal_idx = idx

    def find_fractal(self, idx):
        """
        Method to find candles.
        :return: List (contains one or two fractals. One fractal if either 'Up' or 'Down' is found, two if both are
                    found
        """
        # Indicator variables to specify whether up or down fractals are found
        up, down = 0, 0
        # Current candle high and low
        c3h, c3l = self.candles[idx][1], self.candles[idx][2]
        # Next two candle highs and lows
        c4h, c4l = self.candles[idx+1][1], self.candles[idx+1][2]
        c5h, c5l = self.candles[idx+2][1], self.candles[idx+2][2]
        # Previous two candle highs and lows
        c2h, c2l, c1h, c1l = self.check_left_candles(idx, c3h, c3l)
        # Check for 'Up' fractal
        if (c3h > c4h) & (c3h > c5h):
            if (c2h is not None) & (c1h is not None):
                if (c3h > c2h) & (c3h > c1h):
                    up = 1
        if (c3l < c4l) & (c3l < c5l):
            if (c2l is not None) & (c1l is not None):
                if (c3l < c2l) & (c3l < c1l):
                    down = 1
        return self.make_fractal(up, down, idx)

    def check_left_candles(self, idx, c3h, c3l):
        c2h, c2l, c1h, c1l = None, None, None, None
        # Find high values
        left_candles = copy.deepcopy(self.candles[:idx][::-1][:6])
        parallel_indicator = 0
        while True:
            if ((c2h is not None) & (c1h is not None)) | (len(left_candles) == 0):
                break
            candidate = left_candles.pop(0)[1]
            if candidate != c3h:
                if c2h is None:
                    c2h = candidate
                else:
                    c1h = candidate
            else:
                if parallel_indicator == 1:
                    if c2h is None:
                        pass
                    else:
                        break
                else:
                    if c2h is None:
                        parallel_indicator = 1
        # Find high values
        left_candles = copy.deepcopy(self.candles[:idx][::-1][:6])
        parallel_indicator = 0
        while True:
            if ((c2l is not None) & (c1l is not None)) | (len(left_candles) == 0):
                break
            candidate = left_candles.pop(0)[2]
            if candidate != c3l:
                if c2l is None:
                    c2l = candidate
                else:
                    c1l = candidate
            else:
                if parallel_indicator == 1:
                    if c2l is None:
                        pass
                    else:
                        break
                else:
                    if c2l is None:
                        parallel_indicator = 1
        return c2h, c2l, c1h, c1l

    def make_fractal(self, up, down, idx):
        fractal_up = (self.candles[idx][1], "up", self.candles[idx][4])
        fractal_dn = (self.candles[idx][2], "down", self.candles[idx][4])
        if (up == 1) & (down == 1):
            # If both 'up' and 'down' fractals are found in the same candle
            if len(self.fractals) > 0:
                # If fractals are already present
                if self.fractals[-1][1] == "up":
                    # If the last fractal is Up, set the new candle first as a down fractal and then as an Up fractal
                    return [fractal_dn, fractal_up]
                elif self.fractals[-1][1] == "down":
                    # If the last fractal is Down, set the new candle first as an up fractal and then as a down fractal
                    return [fractal_up, fractal_dn]
            else:
                # If no prior fractals are present
                return [fractal_up, fractal_dn]
        elif up == 1:
            # Only Up fractal is found
            return [fractal_up]
        elif down == 1:
            # Only Down fractal is found
            return [fractal_dn]
        else:
            # No Fractals are found
            return None
