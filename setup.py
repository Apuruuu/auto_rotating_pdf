from setuptools import setup

setup(
    name='openpylivox_pkg',
    version='1.1.0',
    url='https://github.com/ryan-brazeal-ufl/openpylivox',
    author='Ryan Brazeal',
    author_email='ryan.brazeal@ufl.edu',
    packages=['openpylivox'],
    description='Python3 driver for Livox lidar sensors',    
    install_requires=['PyPDF4'],
    python_requires='>=3.8',
)
