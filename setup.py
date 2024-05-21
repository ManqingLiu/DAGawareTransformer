from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='dag-aware-transformer',
    version='1.0.0',
    author='Manqing Liu',
    packages=find_packages(),
    install_requires=requirements,
)