# Copyright (C) 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

try:
    import spaces
except ImportError:
    spaces = None


def with_spaces_gpu(duration: int = 30):
    """Reserve a ZeroGPU GPU for a Gradio event handler when Spaces is installed."""

    def decorator(fn):
        if spaces is None:
            return fn

        return spaces.GPU(duration=duration)(fn)

    return decorator
