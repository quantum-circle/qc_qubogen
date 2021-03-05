# coding: utf-8

"""
    qc_qubogen

    Main incubator Quantum Generator Project(oauth).  # noqa: E501
    
"""


from setuptools import setup, find_packages  # noqa: H301

NAME = "qc_qubogen"
VERSION = "1.0.0"
# To install the library, run the following
#
# python setup.py install
#
# prerequisite: setuptools
# http://pypi.python.org/pypi/setuptools    

with open('requirements.txt') as f:
    requirements = [s.strip() for s in f.readlines()]


setup(
    name=NAME,
    version=VERSION,
    description="qc_qubogen",
    author_email="",
    url="",
    keywords=["qc_qubogen"],
    install_requires=requirements,
    packages=find_packages(),
    long_description="""\
    Main incubator Quantum Generator Project(oauth).  # noqa: E501
    """
)
