from loguru import logger


def catch_exception_as_warning(f):
    def wrapper(*args, **kw):
        try:
            return f(*args, **kw)
        except Exception as exc:
            logger.warning(exc)

    return wrapper
