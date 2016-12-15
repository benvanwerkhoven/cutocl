#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('VERSION') as version_file:
    project_version = version_file.read()

setup(
    name='cutocl',
    version=project_version,
    description="A Python tool for quick and dirty translation of CUDA kernels to OpenCL",
    long_description=readme + '\n\n',
    author="Ben van Werkhoven",
    author_email='b.vanwerkhoven@esciencecenter.nl',
    url='https://github.com/benvanwerkhoven/cutocl',
    packages=[
        'cutocl',
    ],
    entry_points = {
        "console_scripts": ['cutocl = cutocl.cutocl:main']
    },
    package_dir={'cutocl':
                 'cutocl'},
    include_package_data=True,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='cutocl',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
)
