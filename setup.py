# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


setup(
    name='FMLite',
    version='0.0.1',
    description='Pure Python Implementation of Factoriazatin Machine.',
    author='Moriaki Saigusa',
    author_email='moriaki3193@gmail.com',
    url='https://github.com/moriaki3193/FMLite',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'notebooks')),
    test_suite='tests'
)
