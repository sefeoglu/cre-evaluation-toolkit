from setuptools import setup

def parse_requirements(requirements):
    with open(requirements) as f:
        return [l.strip('\n') for l in f if l.strip('\n') and not l.startswith('#')]

reqs = parse_requirements("requirements.txt")

setup(
   name='cre-evaluation-toolkit',
   version='1.0',
   description='A Continual Relation Extraction Evaluator',
   author='Sefika Efeoglu',
   author_email='sefikaefeoglu@gmail.com',
   packages=['cre-evaluation-toolkit'],
   install_requires=reqs,
   classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: MIT',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)