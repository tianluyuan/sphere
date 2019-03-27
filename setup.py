from setuptools import setup, find_packages

setup(
    name='sphere',
    version='0.1.0',
    author='T. Yuan',
    author_email='tyuan@icecube.wisc.edu',
    description='Implementation of Kent (1982) and Bingham-Mardia (1978) distributions on a sphere',
    long_description=open('README.md').read(),
    url='https://github.com/tianluyuan/sphere.git',
    packages=find_packages('./'),
    install_requires=['numpy',
                      'scipy'],
    extras_require={
        'plotting':  ['matplotlib', 'healpy']
    },
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
    )
