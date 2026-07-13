"""Installation script for the ``contact_valid_place_rl`` package."""

from __future__ import annotations

import os

import toml
from setuptools import find_packages, setup


EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

INSTALL_REQUIRES = [
    "psutil",
]

setup(
    name="contact_valid_place_rl",
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    packages=find_packages(include=["contact_valid_place_rl", "contact_valid_place_rl.*"]),
    install_requires=INSTALL_REQUIRES,
    include_package_data=True,
    python_requires=">=3.10",
    zip_safe=False,
)
