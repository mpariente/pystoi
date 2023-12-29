from setuptools import setup
from setuptools import find_packages

with open("README.md", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='pystoi',
    version='0.4.1',
    description='Computes Short Term Objective Intelligibility measure',
    author='Manuel Pariente',
    author_email='pariente.mnl@gmail.com',
    url='https://github.com/mpariente/pystoi',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    install_requires=['numpy', 'scipy'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3'
    ],
    packages=find_packages()
)
