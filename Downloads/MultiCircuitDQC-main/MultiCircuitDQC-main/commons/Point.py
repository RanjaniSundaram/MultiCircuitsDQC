class Point:
    """Point object have x and y parameters as cartesian values of a location"""
    def __init__(self, x, y):
        self._x, self._y = x, y

    def distance(self, other: 'Point'):
        """Return distance between two locations"""
        return ((self._x - other._x) ** 2 + (self._y - other._y) ** 2) ** 0.5

    def __str__(self):
        return "{x},{y}".format(x=self._x, y=self._y)
    
    @property
    def x(self):
        return self._x
    
    @property
    def y(self):
        return self._y
