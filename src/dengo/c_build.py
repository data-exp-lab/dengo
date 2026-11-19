"""Compile dengo-generated C++ sources into plain static/shared libraries
(no Python extension module) -- used for the grid-level C API
(`ChemicalNetwork.write_c_api`) and the Grackle-compatible library
(`dengo.grackle_shim`). Just drives the system C++ compiler directly.
"""
import logging
import os
import shutil
import subprocess

logger = logging.getLogger("dengo")


def _cxx():
    return os.environ.get("CXX", "g++")


def _run(cmd):
    logger.debug("%s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "command failed (%d): %s\n%s%s" % (proc.returncode, " ".join(cmd), proc.stdout, proc.stderr)
        )


def _compile_objects(sources, obj_dir, include_dirs, defines, extra_flags):
    os.makedirs(obj_dir, exist_ok=True)
    objects = []
    for src in sources:
        obj = os.path.join(obj_dir, os.path.splitext(os.path.basename(src))[0] + ".o")
        cmd = [_cxx(), "-x", "c++", "-c", "-O3", "-fno-math-errno", "-fPIC", "-std=c++11",
               src, "-o", obj]
        cmd += ["-I%s" % d for d in include_dirs]
        cmd += ["-D%s" % d for d in defines]
        cmd += list(extra_flags)
        _run(cmd)
        objects.append(obj)
    return objects


def openmp_works():
    """True if the C++ compiler accepts -fopenmp (it doesn't, e.g., with
    stock Xcode clang)."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "t.C")
        with open(src, "w") as f:
            f.write("#include <omp.h>\nint main(){return omp_get_max_threads() > 0 ? 0 : 1;}\n")
        try:
            _run([_cxx(), "-fopenmp", src, "-o", os.path.join(tmp, "t")])
            return True
        except RuntimeError:
            return False


def build_library(name, sources, output_dir, include_dirs=(), defines=(),
                  use_openmp=None, shared=True, static=True, serial_sources=()):
    """Compile `sources` into `output_dir/lib<name>.so` and/or
    `lib<name>.a`. `use_openmp`: True/False to force, None to use it if
    the compiler supports it. `serial_sources` are compiled without
    OpenMP even then -- for code that is only ever called from inside
    another file's parallel region, where its own `omp parallel`
    pragmas would just make libgomp set up (and allocate) a nested team
    on every call. Returns a dict with the paths built and whether
    OpenMP was used."""
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if use_openmp is None:
        use_openmp = openmp_works()
        if not use_openmp:
            logger.warning("Building lib%s without OpenMP (compiler does not "
                           "support -fopenmp); it will run single-threaded.", name)
    omp = ["-fopenmp"] if use_openmp else []
    obj_dir = os.path.join(output_dir, "_obj_%s" % name)
    parallel = [s for s in sources if s not in serial_sources]
    serial = [s for s in sources if s in serial_sources]
    objects = _compile_objects(parallel, obj_dir, include_dirs, defines, omp)
    objects += _compile_objects(serial, obj_dir, include_dirs, defines, [])

    result = {"openmp": bool(use_openmp)}
    if shared:
        so = os.path.join(output_dir, "lib%s.so" % name)
        # SONAME, so binaries record "lib<name>.so" (not this path) and
        # LD_LIBRARY_PATH / a moved install can select the library.
        _run([_cxx(), "-shared", "-Wl,-soname,lib%s.so" % name, "-o", so] + objects + omp)
        result["shared"] = so
    if static:
        a = os.path.join(output_dir, "lib%s.a" % name)
        if os.path.exists(a):
            os.remove(a)
        _run([os.environ.get("AR", "ar"), "rcs", a] + objects)
        result["static"] = a
    shutil.rmtree(obj_dir, ignore_errors=True)
    return result
