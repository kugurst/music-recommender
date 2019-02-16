import sys
import platform


# https://stackoverflow.com/a/22820100
def is_windows():
    return any(platform.win32_ver())
