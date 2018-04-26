from setuptools import setup, find_packages

setup(
    name='demo101mhu',
    version='0.2',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='An example python package',
    long_description=open('README.md').read(),
    install_requires=['numpy','pandas','scipy','matplotlib','seaborn','scikit-learn','statsmodels'],
    url='https://github.com/mhu10/demo101mhu',
    author='mhu10',
    author_email='mhu10@jhmi.edu'
)
