from setuptools import setup

setup(name='MSMRD',
      version='0.1',
      description='Reaction diffusion with MSM interaction',
      author='Manuel Dibak',
      author_email='manuel.dibak@fu-berlin.de',
      url='',
      packages=['MSMRD', 'MSMRD.potentials', 'MSMRD.integrators'],
      test_suite='nose.collector',
      tests_require=['nose']
      )
