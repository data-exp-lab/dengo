"""
Compile a dengo-generated Cython solver into an importable extension.

No Makefile, no pyximport magic: this drives `Cython.Build.cythonize` and
an in-process `setuptools` `build_ext` directly, so "define a network ->
generate a solver -> compile it -> call it" is one Python call and works
under a plain `uv run`. The only requirements are a C++ compiler and
Cython -- no HDF5, no SUNDIALS.

Every cell in a batch is independent (no cross-cell terms anywhere in
this network), so the generated solver's per-cell loops are OpenMP
parallel-for regions. `-fopenmp` isn't universally available (notably:
stock Xcode clang on macOS doesn't ship it), so this tries an OpenMP
build first and falls back to a serial one automatically if that fails
to compile -- either way you get a working solver back.
"""

import glob
import importlib.util
import logging
import os
import shutil

logger = logging.getLogger("dengo")


def _build_extension(output_dir, module_name, pyx_path, solver_name, openmp_args):
    import numpy as np
    from Cython.Build import cythonize
    from setuptools import Extension
    from setuptools.dist import Distribution

    ext = Extension(
        name=module_name,
        sources=[
            pyx_path,
            os.path.join(output_dir, "%s_solver.C" % solver_name),
            os.path.join(output_dir, "BE_chem_solve.C"),
        ],
        include_dirs=[output_dir, np.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-w"] + openmp_args,
        extra_link_args=list(openmp_args),
    )
    build_tmp = os.path.join(output_dir, "_build")
    ext_modules = cythonize(
        [ext],
        compiler_directives={"language_level": "3"},
        force=True,
    )

    dist = Distribution({"name": module_name, "ext_modules": ext_modules})
    # NOT --inplace: for a flat (non-package) extension name, distutils'
    # in-place copy step resolves relative to the current working
    # directory rather than build_lib, regardless of where the sources
    # live -- it was silently dropping a duplicate .so at the caller's
    # cwd on top of the correct one in output_dir. Plain --build-lib
    # (found below via glob) is sufficient and doesn't have that problem.
    dist.script_args = [
        "build_ext",
        "--build-lib", output_dir,
        "--build-temp", build_tmp,
    ]
    dist.parse_command_line()
    dist.run_commands()


def build_solver(output_dir, solver_name, use_openmp=None):
    """Compile the `<solver_name>_solver_run.pyx` written by
    `ChemicalNetwork.write_cython_solver()` in `output_dir`, and return the
    imported extension module (exposing `run_<solver_name>(...)`).

    `use_openmp`: True/False to force OpenMP on/off, or None (default) to
    try OpenMP and silently fall back to a serial build if that fails to
    compile on this platform/toolchain.
    """
    output_dir = os.path.abspath(output_dir)
    module_name = "%s_solver_run" % solver_name
    pyx_path = os.path.join(output_dir, "%s.pyx" % module_name)
    if not os.path.exists(pyx_path):
        raise FileNotFoundError(
            "%s not found -- call ChemicalNetwork.write_cython_solver() "
            "first to generate it" % pyx_path
        )

    build_tmp = os.path.join(output_dir, "_build")

    attempts = [["-fopenmp"]] if use_openmp in (None, True) else []
    if use_openmp in (None, False):
        attempts.append([])

    last_error = None
    for openmp_args in attempts:
        try:
            _build_extension(output_dir, module_name, pyx_path, solver_name, openmp_args)
            if not openmp_args and use_openmp is None:
                logger.warning(
                    "Building %s without OpenMP -- an OpenMP build was "
                    "tried first and failed to compile on this "
                    "platform/toolchain; the solver still works, just "
                    "single-threaded.", solver_name,
                )
            last_error = None
            break
        except Exception as exc:  # noqa: BLE001 -- compiler errors vary by platform
            last_error = exc
            shutil.rmtree(build_tmp, ignore_errors=True)
    if last_error is not None:
        raise last_error

    matches = sorted(
        glob.glob(os.path.join(output_dir, module_name + "*.so"))
        + glob.glob(os.path.join(output_dir, module_name + "*.pyd"))
    )
    if not matches:
        raise RuntimeError(
            "Build finished but no extension module was found for "
            "%s in %s" % (module_name, output_dir)
        )
    so_path = matches[-1]

    spec = importlib.util.spec_from_file_location(module_name, so_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
