import importlib.util
import os
import sys

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

HERE = os.path.dirname(os.path.abspath(__file__))


def _version_module():
    """Load netqmpi/version.py on its own, without importing the package."""
    path = os.path.join(HERE, 'netqmpi', 'version.py')
    spec = importlib.util.spec_from_file_location('_netqmpi_version', path)
    module = importlib.util.module_from_spec(spec)
    # Registered first: dataclasses looks its defining module up there.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_build_info(package_dir, version_string):
    """
    Freeze the git tag and commit of this checkout into ``_build_info.py``.

    Written into the build or release tree, never into the sources. Without a
    git checkout — building from an sdist — nothing is written, and the
    ``_build_info.py`` the sdist already carries is the one that ships.

    Args:
        package_dir: Directory of the ``netqmpi`` package in the tree being
            built.
        version_string: The release being built.
    """
    version = _version_module()
    info = version.read_git(HERE)
    if info is None:
        return
    info['version'] = version_string
    target = os.path.join(package_dir, version.BUILD_INFO_MODULE + '.py')
    os.makedirs(package_dir, exist_ok=True)
    if os.path.exists(target):
        os.remove(target)           # may be a hard link into the sources
    with open(target, 'w', encoding='utf-8') as handle:
        handle.write(version.render_build_info(info))


class BuildPyWithBuildInfo(build_py):
    """``build_py`` that also records which commit the build came from."""

    def run(self):
        super().run()
        if not self.dry_run:
            _write_build_info(os.path.join(self.build_lib, 'netqmpi'),
                              self.distribution.get_version())


class SdistWithBuildInfo(sdist):
    """``sdist`` that also records which commit the archive came from."""

    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        _write_build_info(os.path.join(base_dir, 'netqmpi'),
                          self.distribution.get_version())


setup(
    name='netqmpi',
    version='0.3.1',
    cmdclass={
        'build_py': BuildPyWithBuildInfo,
        'sdist': SdistWithBuildInfo,
    },
    entry_points={
        'console_scripts': [
            'netqmpi=netqmpi.runtime.cli:main',
        ],
    },
    packages=find_packages(),
    install_requires=[
        'pyyaml>=5.1',      # --config parsing
    ],
    extras_require={
        'test': ['pytest>=7.0'],
    },
    author='F. Javier Cardama',
    author_email='javier.cardama@usc.es',
    description='A high-level abstraction layer similar to MPI for distributed quantum programming over NetQASM.',
    url='https://github.com/NetQIR/netqmpi',  # Replace with your project's URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
