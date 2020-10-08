from setuptools import setup, find_packages

setup(
        name="adacvar",
        version="0.0.1",
        author="Sebastian Curi",
        author_email="sebascuri@gmail.com",
        license="MIT",
        packages=find_packages(exclude=['docs']),
        python_requires=">=3.6.0",
        install_requires=['numpy>=1.0,<2',
                          'scipy>=0.19.1,<1.4.0',
                          'tqdm>=4.0,<5.0',
                          'PyYAML>=5.1',
                          'scikit-learn>=0.20',
                          'torch>=1.1.0',
                          'torchvision>=0.2',
                          'matplotlib>=3.0',
                          'Pillow==5.4.1',
                          ],
        extras_require={
            'test': [
                'pytest>=4.6,<4.7',
                'flake8==3.7.7',
                'pydocstyle==3.0.0',
                'pytest_cov>=2.7,<3',
                "black>=19.10b0",
                "isort>=5.0.0",
            ],
        },
        classifiers=[
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 3 - Alpha',

            # Indicate who your project is intended for
            'Intended Audience :: Developers',
            'Topic :: Software Development :: Build Tools',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7'
        ],
)
