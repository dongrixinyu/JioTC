# -*- coding=utf-8 -*-

import os

from setuptools import setup, find_packages


DIR_PATH = os.path.dirname(os.path.abspath(__file__))
LONGDOC = '''
一个文本分类的模型框架，方便提供词向量，bert finetune，词典，规则，部署优化等接口功能。
'''

__name__ = 'jiotc'
__author__ = "cuiguoer"
__copyright__ = "Copyright 2020, dongrixinyu"
__credits__ = []
__license__ = "Apache License 2.0"
__maintainer__ = "dongrixinyu"
__email__ = "dongrixinyu.89@163.com"
__url__ = ''
__description__ = LONGDOC  # .split('安装\n```')[0]



#with open(os.path.join(DIR_PATH, 'requirements.txt'), 
#          'r', encoding='utf-8') as f:
#    requirements = f.readlines()

setup(name=__name__,
      version='0.1.0',
      url=__url__,
      author=__author__,
      author_email=__email__,
      description=__description__,
      long_description=LONGDOC,
      license=__license__,
      py_modules=[],
      packages=find_packages(),
      include_package_data=True,
      #install_requires=requirements,
      entry_points={
          'console_scripts': [
              # 'scheduler_start = algorithm_platform.scheduler.server: start',
          ]
      },
      test_suite='nose.collector',
      tests_require=['nose'])

