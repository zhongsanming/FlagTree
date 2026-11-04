"""triton-ascend spec-level hooks.

The modules under ``third_party/ascend/spec/triton`` are used as path
overrides for ``triton.compiler``, ``triton.language`` and ``triton.runtime``.
This package only hosts small call/function-level hooks needed outside those
packages.
"""

__all__ = ["language_extend_globals"]


def language_extend_globals(globals_dict):
    # Expose the TensorView frontend types on ``triton.language``. The actual
    # view constructors are provided through ``tl.ext`` by
    # triton.backends.ascend.language_ext.
    from triton.language.core import scatter, tensor_view, tensor_view_type

    for name, value in (
        ("scatter", scatter),
        ("tensor_view", tensor_view),
        ("tensor_view_type", tensor_view_type),
    ):
        globals_dict.setdefault(name, value)
