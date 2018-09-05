import setuptools

if __name__ == '__main__':
    setuptools.setup(
        name='bremen',
        author='Yusuke Nakashima',
        author_email='nakashima.y.ac@m.titech.ac.jp',
        version='0.1.0',
        packages=setuptools.find_packages(),
        install_requires=[
            'matplotlib',
            'numpy',
            'chainer==1.24',
            'scipy',
        ],
        entry_points={
            'console_scripts': [
                'albrecht = bremen.albrecht:main',
                'giselle = parameter_search.parameter_search:main',
            ]
        }
    )
