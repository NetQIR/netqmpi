from setuptools import setup, find_packages

setup(
    name='netqmpi',
    version='0.1.0',
    entry_points={
        'console_scripts': [
            'netqmpi=netqmpi.runtime.cli:main', 
        ],
    },
    packages=find_packages(),
    author='F. Javier Cardama',
    author_email='javier.cardama@usc.es',
    description='A high-level abstraction layer similar to MPI for distributed quantum programming over NetQASM.',
    url='https://github.com/NetQIR/net-qmpi',  # Replace with your project's URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)