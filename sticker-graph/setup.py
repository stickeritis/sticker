from setuptools import setup

setup(name='sticker-graph',
      version='0.11.0',
      description='Graph construction utilities for sticker',
      url='https://github.com/danieldk/sticker',
      author='Daniël de Kok',
      author_email='me@danieldk.eu',
      license='BlueOak-1.0.0',
      tests_require=[
          'numpy',
          'tensorflow == 1.15.4',
          'toml',
      ],
      install_requires=[
          'tensorflow == 1.15.4',
          'toml',
      ],
      packages=['sticker_graph'],
      scripts=[
          'sticker-write-conv-graph',
          'sticker-write-rnn-graph',
          'sticker-write-transformer-graph'
      ])
