import math
from collections import defaultdict

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

def combine(c, x1, d, x2):
    """linear combination of two sparse vectors: c(x1) + d(x2)
    """
    out = defaultdict(float)
    for f in set(x1.keys()) | set(x2.keys()):
        out[f] = (c * x1[f]) + (d * x2[f])
    return out

def discretizeLocation(x, y):
    """converts continuous coordinates in R^2 to discrete location measurement 

    does so by converting game board to grid of 20x20 pixel squares, then
      gives the index of the square that (x, y) is in
    """
    entries_in_row = SCREEN_SIZE[0] / 20
    x_grid = x / 10
    y_grid = y / 10
    return x_grid + y_grid * (SCREEN_SIZE[0] / 20)

def discretizeAngle(vec):
    """buckets the continuous angle of a vector into one of 16 discrete angle categories
    """
    return int(utils.angle(vec) / 10)

def set_bit(bv, i):
    return bv | (1 << i)

def serializeBinaryVector(vec, use_bricks=False):
    if use_bricks:
        return '|'.join(sorted(vec.keys()))
    else:
        return '|'.join(k for k in sorted(vec.keys()) if 'brick' not in k)

def serializeList(l):
    return tuple(sorted(l))

def deserializeAction(a):
    return list(a)

def allSame(l):
    return all(x == l[0] for x in l)
