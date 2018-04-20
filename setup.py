from setuptools import setup

setup(
    name='pystoi',
    version='0.0.1',
    description='Computes Short Term Objective Intelligibility measure',
    author='Manuel Pariente',
    url='https://github.com/mpariente/pystoi',
    license='MIT',
    packages=['pystoi'],
    install_requires=['numpy', 'scipy', 'librosa']
)
