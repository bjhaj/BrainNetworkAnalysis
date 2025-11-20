from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='neurograph',
    version='2.0.0',
    description='Benchmarks for Graph Machine Learning in Brain Connectomics',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Anwar Said',
    author_email='anwar.said@vanderbilt.edu',
    url='https://github.com/Anwar-Said/NeuroGraph',
    packages=find_packages(include=['neurograph', 'neurograph.*']),
    package_data={
        'neurograph': ['data/reference/*'],
    },
    install_requires=[
        'torch>=2.0.0',
        'torch_geometric>=2.3.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
        'scipy>=1.10.0',
        'networkx>=3.0',
        'tqdm>=4.65.0',
        'boto3',
        'nilearn',
        'nibabel',
        'sphinx_rtd_theme',
    ],
    extras_require={
        'baselines': [
            'grakel>=0.1.8',  # For WL kernel baselines
        ],
        'dev': [
            'pytest>=7.0',
            'black>=23.0',
            'flake8>=6.0',
        ],
    },
    keywords=['python', 'neuroimaging', 'graph machine learning', 'brain connectomics', 'GNN'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
    python_requires='>=3.8',
)
