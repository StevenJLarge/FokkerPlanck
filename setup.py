from setuptools import find_packages, setup

# README contents
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='fokker-planck',
    packages=find_packages(),
    version='1.0.17',
    long_description=long_description,
    long_description_content_type='text/markdown',
    description='''
        A Fokker-Planck integrator in python
    ''',
    author='Steven Large',
    author_email='stevelarge7@gmail.com',
    license='MIT',
    install_requires=[
        "matplotlib",
        "numpy",
        "scipy",
        "statsmodels"
    ],
    project_urls={
        'Source': 'https://github.com/StevenJLarge/FokkerPlanck'
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics"
    ]

)
