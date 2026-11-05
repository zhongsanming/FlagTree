# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Ascend backend setup hooks for the upstream triton-ascend 3.6 port."""

import os
import platform
import shutil
import subprocess
import sys

# setup_helper loads this file with importlib under the synthetic module name
# "module"; make its sibling modules importable by name.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Upstream triton-ascend 3.6 pins LLVM f6ded0be and applies this patch before
# building. The resulting MLIR has extra bufferization ops (to_memref,
# to_tensor extensions) used by AscendNPU-IR.
LLVM_REV = "f6ded0be"
LLVM_PATCH_ID = "3f5016d1"
LLVM_ARTIFACT_ROOT = "https://triton-ascend-artifacts.obs.myhuaweicloud.com/llvm-builds"

ASCENDNPU_IR_NAME = "AscendNPU-IR"
ASCENDNPU_IR_URL = "https://gitcode.com/Ascend/AscendNPU-IR.git"
ASCENDNPU_IR_COMMIT = "bb13ef6120526aa3df6f950006af5ec0e585d2f6"

# Proton/FlagPrism are not required to compile or run the Ascend backend. Keep
# them opt-in; the NVIDIA dependency download is skipped entirely in setup.py.
os.environ.setdefault("TRITON_BUILD_PROTON", "OFF")
os.environ.setdefault("TRITON_BUILD_FLAGPRISM", "OFF")


def precompile_hook_ascend(*args, **kwargs):
    """Fetch the FLIR revision required by the Ascend CommonIR lowering."""
    from tools import DownloadManager, Module, flagtree_configs

    downloader = DownloadManager()
    flir = Module(
        name="flir-ascend",
        url="https://github.com/kateyijian/flir.git",
        commit_id="482b60b881219cab7df987f005b9d8bf5c97cbe5",
        dst_path=os.path.join(flagtree_configs.flagtree_submodule_dir, "flir"),
    )
    downloader.download(module=flir, required=True)
    return kwargs.get("default_backends")


def install_extension(*args, **kwargs):
    """Copy the Ascend MLIR tools next to libtriton.so.

    The runtime looks for ``triton-mlir-opt`` in ``triton/_C`` before PATH.
    ``triton-opt`` is optional but copied when the core bin target was built.
    """
    build_ext = kwargs.get("build_ext")
    if build_ext is None:
        return
    try:
        ext = build_ext.extensions[0]
        extdir = os.path.abspath(os.path.dirname(build_ext.get_ext_fullpath(ext.path)))
    except (AttributeError, IndexError):
        return
    try:
        from python.build_helpers import get_cmake_dir
        cmake_dir = get_cmake_dir()
    except Exception:
        cmake_dir = getattr(build_ext, "build_temp", None)
    if not cmake_dir:
        return

    for rel_src, name in (
        ("third_party/ascend/bin/triton-mlir-opt", "triton-mlir-opt"),
        ("bin/triton-opt", "triton-opt"),
    ):
        src = os.path.join(cmake_dir, rel_src)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(extdir, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        if platform.system() != "Windows":
            os.chmod(dst, 0o755)
            try:
                subprocess.run(["strip", "--strip-all", dst], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                pass
        print(f"[ascend] copied {name} to {dst}")


def _is_linux_os(os_id: str) -> bool:
    try:
        with open("/etc/os-release") as f:
            return f'ID="{os_id}"' in f.read()
    except OSError:
        return False


def _llvm_system_suffix() -> str | None:
    system = platform.system()
    try:
        arch = {"x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}[platform.machine()]
    except KeyError:
        arch = platform.machine()

    if system == "Darwin":
        return f"macos-{arch}"
    if system != "Linux":
        return None

    if arch == "arm64":
        return "almalinux-arm64" if _is_linux_os("almalinux") else "ubuntu-arm64"
    if arch == "x64":
        try:
            major, minor = platform.libc_ver()[1].split(".")[:2]
            vglibc = int(major) * 100 + int(minor)
        except (ValueError, IndexError):
            vglibc = 0
        return "ubuntu-x64" if vglibc > 228 else "almalinux-x64"
    return None


def _llvm_package() -> tuple[str, str] | None:
    suffix = _llvm_system_suffix()
    if suffix is None:
        return None
    name = f"llvm-{LLVM_REV}-{LLVM_PATCH_ID}-{suffix}"
    return name, f"{LLVM_ARTIFACT_ROOT}/{name}.tar.gz"


def register_cache(cache, flagtree_backend, check_env, set_llvm_env):
    """Download the triton-ascend-patched LLVM unless LLVM_SYSPATH is set."""
    package = _llvm_package()
    if package is None:
        return
    name, url = package
    cache.store(
        file=name,
        condition=("ascend" == flagtree_backend),
        url=url,
        pre_hook=lambda: check_env("LLVM_SYSPATH"),
        post_hook=set_llvm_env,
    )


def get_backend_cmake_args(build_ext=None, **kwargs):
    return [
        "-DLLVM_MAJOR_VERSION_22_COMPATIBLE=ON",
        "-DTRITON_BUILD_TD=OFF",
        # Start lean. The in-process cost model can be enabled after the port
        # builds and its unit tests pass.
        "-DTRITON_ASCEND_ENABLE_COSTMODEL_INPROC=OFF",
    ]


def get_package_data(backends):
    """Package the Ascend bitcode, native launcher source, and BiSheng tools."""
    return {
        "triton.backends.ascend": [
            "lib/*.bc",
            "npu_utils.cpp",
            "*.json",
            "bishengir/bin/*",
            "bishengir/lib/*",
            "bishengir/lib/**/*",
        ],
    }


def post_install():
    """No-op hook kept for symmetry with other backends."""
    return None


def skip_package_dir(package):
    # The bishengir payload is not a Python package; it is copied as package
    # data by get_package_data().
    return package == "triton.backends.ascend.bishengir"



