import math


def dotProduct(a, b):
    """ dot product of two vectors """
    return sum(x * y for (x, y) in zip(a, b))

def magnitude(a):
    """ magnitude of vector """
    return math.sqrt(dotProduct(a, a))

def normalize(vec):
    """ normalize vec to unit vector:  ||vec'|| = 1 """
    return [x * 1.0 / magnitude(vec) for x in vec]

def angle(vec):
    """ angle of vector in degrees (compared with [0, 1]) """
    a = angleBetween([0, 1], vec)
    return 360 - a if vec[0] < 0 else a

def angleBetween(a, b):
    """angle between two vectors"""
    a = normalize(a)
    b = normalize(b)
    return math.degrees(math.acos(dotProduct(a, b) / (magnitude(a) * magnitude(b))))
