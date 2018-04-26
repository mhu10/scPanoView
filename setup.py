from setuptools import setup, find_packages

setup(
    name='scPanoView',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='An example python package',
    long_description=open('README.md').read(),
    install_requires=['numpy','pandas','scipy','matplotlib','seaborn','scikit-learn','statsmodels'],
    url='https://github.com/mhu10/scPanoView',
    author='mhu10',
    author_email='mhu10@jhmi.edu'
)
