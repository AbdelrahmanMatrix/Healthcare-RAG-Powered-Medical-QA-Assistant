# api.routes package
# Must use a relative import — absolute import causes circular ImportError at startup.
from . import query   # noqa: F401
__all__ = ["query"]
