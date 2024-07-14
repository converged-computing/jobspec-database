from setuptools import setup

REQUIRED = [
    "matplotlib",
    "pyaml",
    "optuna==3.4.0",
    "smac==2.0.2",
    "botorch==0.9.4",
]


setup(name="dhexp", packages=["dhexp"], install_requires=REQUIRED)
