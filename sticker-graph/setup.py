from setuptools import setup

setup(name='sticker-graph',
      version='0.4.0',
      description='Graph construction utilities for sticker',
      url='https://github.com/danieldk/sticker',
      author='Daniël de Kok',
      author_email='me@danieldk.eu',
      license='Apache2',
      tests_require=[
          'numpy',
          'tensorflow == 1.13.1',
          'toml',
      ],
      install_requires=[
          'tensorflow == 1.13.1',
          'toml',
      ],
      packages=['sticker_graph'],
      scripts=[
          'sticker-write-conv-graph',
          'sticker-write-rnn-graph'
      ])
