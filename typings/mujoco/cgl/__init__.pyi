"""
An Apple CGL context for offscreen rendering on macOS.
"""
from __future__ import annotations
import atexit as atexit
import ctypes as ctypes
from mujoco.cgl.cgl import CGLOpenGLProfile as _PROFILE
from mujoco.cgl.cgl import CGLPixelFormatAttribute as _ATTRIB
import os as os
from . import cgl
__all__: list[str] = ['GLContext', 'atexit', 'cgl', 'ctypes', 'os']
class GLContext:
    """
    An EGL context for headless accelerated OpenGL rendering on GPU devices.
    """
    def __del__(self):
        ...
    def __init__(self, max_width, max_height):
        ...
    def free(self):
        """
        Frees resources associated with this context.
        """
    def make_current(self):
        ...
