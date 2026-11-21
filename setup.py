from setuptools import find_packages, setup

def parse_requirements(requirements):
    with open(requirements) as f:
        return [l.strip('\n') for l in f if l.strip('\n') and not l.startswith('#')]

reqs = parse_requirements("requirements.txt")

setup(
   name='CREvaluator',
    version='0.1.0',
    description='Metrics for evaluating continual-learning experiments',
   author='Sefika Efeoglu',
   author_email='sefikaefeoglu@gmail.com',
    packages=find_packages(),
    python_requires='>=3.10',
   install_requires=reqs,
   classifiers=[
       'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
       'License :: OSI Approved :: MIT License',
       'Operating System :: OS Independent',
       'Programming Language :: Python :: 3',
       'Programming Language :: Python :: 3.10',
       'Programming Language :: Python :: 3.11',
       'Programming Language :: Python :: 3.12',
       'Programming Language :: Python :: 3.13',
    ],
)