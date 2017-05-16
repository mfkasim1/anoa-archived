from setuptools import setup, find_packages

setup(
    name='anoa',
    version='0.1.1a',
    description='A differentiable programming for scientists',
    url='https://github.com/mfkasim91/anoa',
    author='mfkasim91',
    author_email='firman.kasim@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        "numpy>=1.12",
        "scipy>=0.15"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        
        "Programming Language :: Python :: 2.7"
    ],
    keywords="autodiff compressed-sensing differentiable-programming optimization",
    zip_safe=False)
