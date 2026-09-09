"""Build the primordial network/solver for use inside .grackle_compare's
own venv (Python 3.12, to match gracklepy's wheel) -- a separate build
directory from examples/_primordial_network_build, since a compiled
Cython extension is tied to the interpreter ABI it was built for and
that directory's .so is built for whatever Python the main project env
uses. Network definition itself lives in dengo.primordial_network (also
used by examples/primordial_network.py and dengo.grackle_compat)."""
import os

from dengo.primordial_network import build_network as _build_network
from dengo.primordial_network import build_solver as _build_solver

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_dengo_build")


def build_network():
    return _build_network()


def build_solver(network):
    return _build_solver(network, OUTPUT_DIR)
