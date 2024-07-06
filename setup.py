# Data Modeling Python Library
# Created by Christian Garcia

# Import Required libraries for setting up packages
from setuptools import setup, find_packages

# Display markdown description in repo/main page 
with open('README.md', 'r', encoding='utf-8') as file:
    description = file.read()

# Set up package, versions and dependencies
setup(
        name='data-model-patterns',
        version='3.1.0',
        packages=find_packages(),
        install_requires=[
            'matplotlib>=3.4.3',
            'numpy>=1.21.0',
            'pandas>=1.1.3',
            'scipy>=1.7.0',
            'scikit-learn>=0.24.0',
            'mlxtend>=0.17.0'
            ],
        entry_points={
            'console_scripts': [
                'DataModeling = DataModeling.__main__:main'
                ]
            },
        author='Christian Garcia',
        author_email='iyaniyan03112003@gmail.com',
        description='A mining model is created by applying an algorithm to data, but it is more than an algorithm or a metadata container: it is a set of data, statistics, and patterns that can be applied to new data to generate predictions and make inferences about relationships.',
        long_description=description,
        long_description_content_type='text/markdown',
        url='https://github.com/christiangarcia0311/data-model-patterns',
        license='MIT',
        classifiers= [
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Topic :: Software Development :: Libraries :: Python Modules',
            ],
        )