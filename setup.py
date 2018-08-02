from setuptools import setup, find_packages

setup(
    name='scPanoView',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='A single-cell clustering algorithm',
    long_description=open('README.md').read(),
    install_requires=['numpy','pandas','scipy','matplotlib','seaborn','scikit-learn','statsmodels'],
    url='https://github.com/mhu10',
    author='Ming-Wen Hu & Jiang Qian',
    author_email='mhu10@jhmi.edu'
)
