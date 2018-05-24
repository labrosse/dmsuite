from setuptools import setup

with open('README.rst') as rdm:
    README = rdm.read()

DEPENDENCIES = [
    'numpy>=1.12',
    'scipy>=1.0',
]

setup(
    name='dmsuite',
    use_scm_version=True,

    description='Differenciation matrices',
    long_description=README,

    url='https://github.com/labrosse/dmsuite',
    author='Adrien Morison, Stéphane Labrosse',
    author_email='stephane.labrosse@ens-lyon.fr',

    license='GPLv2',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    py_modules=['dmsuite'],
    install_requires=DEPENDENCIES,
)
