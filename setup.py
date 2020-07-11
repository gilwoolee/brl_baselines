import setuptools

extras = {
    'mpi': [
        'mpi4py'
    ]
}

setuptools.setup(name='brl_baselines',
        version='1.0',
        description='Bayesian RL Models',
        author='Gilwoo Lee',
        author_email='gilwoo301@gmail.com',
        license='',
        packages=setuptools.find_packages(),
        install_requires=['gym','numpy', 'baselines'],
        extras_require=extras,
        )
