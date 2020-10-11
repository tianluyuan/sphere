from setuptools import setup, find_packages

setup(
    name='fb8',
    version='0.2.3',
    author='T. Yuan',
    author_email='tyuan@icecube.wisc.edu',
    description='Implementation of FB8, a generalization of the Kent (1982) and Bingham-Mardia (1978) distributions on a sphere',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/tianluyuan/sphere.git',
    packages=find_packages('./'),
    install_requires=['numpy',
                      'scipy'],
    extras_require={
        'plotting':  ['matplotlib', 'healpy']
    },
    python_requires='>=2.7',
    license=open('LICENSE').readline().split()[0],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        ],
    )
