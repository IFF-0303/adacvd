#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="adacvd",
    version="0.0.1",
    description="Code for the development of AdaCVD",
    author="Frederike Lübeck",
    packages=find_packages(include=['config', 'adacvd', 'exploration']),
    url="",
)
