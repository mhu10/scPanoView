from setuptools import setup, find_packages

setup(
    name='scPanoView',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='A single-cell clustering algorithm',
    long_description=open('README.md').read(),
    install_requires=['numpy>=1.13','pandas>=0.20','scipy>=0.19','matplotlib','seaborn>=0.8','scikit-learn>=0.19','statsmodels>=0.8'],
    url='https://github.com/mhu10/scPanoView',
    author='Ming-Wen Hu & Jiang Qian',
    author_email='mhu10@jhmi.edu,jiang.qian@jhmi.edu'
)
