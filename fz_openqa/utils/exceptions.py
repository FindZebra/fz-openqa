from functools import wraps

from loguru import logger


def catch_exception_as_warning(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.exception(exc)

    return wrapper
