try:
    from .builder import build_loss
    from .losses import *  # noqa: F401,F403
except ImportError:
    build_loss = None

__all__ = ['build_loss']
