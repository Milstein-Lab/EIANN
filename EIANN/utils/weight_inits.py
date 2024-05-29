import math


def scaled_kaiming_init(data, fan_in, scale=1):
    kaiming_bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    data.uniform_(-scale * kaiming_bound, scale * kaiming_bound)


def half_kaiming_init(data, fan_in, scale=1, bounds=None):
    kaiming_bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    if bounds is None:
        raise RuntimeError('half_kaiming_init: bounds should be either >=0 or <=0: %s' % str(bounds))
    if bounds[0] is not None and bounds[0] >= 0:
        data.uniform_(bounds[0], scale * kaiming_bound)
    elif bounds[1] is not None and bounds[1] <= 0:
        data.uniform_(-scale * kaiming_bound, bounds[1])
    else:
        raise RuntimeError('half_kaiming_init: bounds should be either >=0 or <=0: %s' % str(bounds))

