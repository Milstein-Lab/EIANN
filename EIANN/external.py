def register_external(name, func):
    """
    Enables EIANN to look up external custom user functions by str name.
    :param name: str
    :param func: callable
    """
    if not isinstance(name, str) and callable(func):
        raise Exception('register_external: requires str name of function and imported callable; received (%s; %s)' %
                        (name, func))
    globals()[name] = func


def is_registered_external(name):
    return name in globals()