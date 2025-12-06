from contextvars import ContextVar

ctx_origin = ContextVar("origin")


def get_origin():
    return ctx_origin.get()


def set_origin(origin: str):
    ctx_origin.set(origin)
