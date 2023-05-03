# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 11:39:09 2022

@author: jlubbers
"""

from setuptools import setup, find_packages

from os import path

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, 'source', 'kigusiq', '_version.py'), encoding='utf-8') as f:
    exec(f.read())
    
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name = 'kigusiq',
      version = __version__,
      author = 'Jordan',
      author_email = 'jelubber@gmail.com',
      description = 'kiqusiq',
      long_description = long_description,
      long_description_content_type = 'text/markdown',
      url = 'https://github.com/jlubbersgeo/kigusiq',
      package_dir = {'':'source'},
      packages = find_packages(where = 'source'),
      install_requires = [
          'pandas',
          'panel',
          'holoviews',
          'numpy',
          'matplotlib',
          'seaborn',
          'mendeleev',
          'scikit_learn',

          ],
      classifiers = [
          'Programming Language :: Python :: 3',
          'Operating System :: OS Independent'
          ],
      python_requires = '>=3.7'
      
      )
