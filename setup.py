#!/usr/bin/env python

import setuptools
from distutils.core import setup

setup(name='variabilityFMGF',
      version='1.0',
      description='Package to utilize variability field map and variability gradient flow frameworks',
      author='Jae Wook Cho',
      author_email='jae7cho@gmail.com',
      url='https://github.com/jae7cho/reliability_fieldMap_gradientFlow',
      packages=setuptools.find_packages(),
      include_package_data=True,
      package_data={'': [
                        'tutorial/example_data/tutorial_data.npy',
                        'misc/cmaps/gradientFlowCmaps.npy'
                        ]
                    }
     )