from setuptools import setup
from setuptools.extension import Extension

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = { }
extensions = [ ]

if use_cython:
    extensions += [
            Extension("trajectoryTools", [ "MSMRD/trajectories/trajectoryTools.pyx" ]),
            Extension("trajectoryReconstructor", [ "MSMRD/trajectories/trajectoryReconstructor.pyx" ])
        ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    extensions += [
        Extension("trajectoryTools", [ "MSMRD/trajectories/trajectoryTools.pyx" ]),
        Extension("trajectoryReconstructor", [ "MSMRD/trajectories/trajectoryReconstructor.pyx" ])
    ]

setup(name='MSMRD',
      version='0.1',
      description='Reaction diffusion with MSM interaction',
      author='Manuel Dibak',
      author_email='manuel.dibak@fu-berlin.de',
      url='',
      packages=['MSMRD', 'MSMRD.analysis', 'MSMRD.potentials', 'MSMRD.integrators', 'MSMRD.trajectories', 'MSMRD.discretization'],
      test_suite='nose.collector',
      tests_require=['nose'],
      cmdclass = cmdclass,
      ext_package='MSMRD.trajectories',
      ext_modules=extensions
      )
