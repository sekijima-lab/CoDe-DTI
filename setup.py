import setuptools
from pathlib import Path

if __name__ == '__main__':
    setuptools.setup(
        name='bremen',
        author='Yusuke Nakashima',
        author_email='nakashima.y.ac@m.titech.ac.jp',
        version='0.2.0',
        python_requires='>=3.12,<3.13',
        packages=setuptools.find_packages(),
        install_requires=Path(__file__).with_name('requirements-lock.txt').read_text().splitlines(),
        entry_points={
            'console_scripts': [
                'albrecht = bremen.albrecht:main',
                'giselle = parameter_search.parameter_search:main',
            ]
        }
    )
