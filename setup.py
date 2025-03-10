from setuptools import setup
def parse_requirements(requirements):
    with open(requirements) as f:
        return [l.strip('\n') for l in f if l.strip('\n') and not l.startswith('#')]

reqs = parse_requirements("requirements.txt")

setup(
   name='CREvaluator',
   version='1.0',
   description='A Continual Relation Extraction Evaluator',
   author='Sefika Efeoglu',
   author_email='sefikaefeoglu@gmail.com',
   packages=['CREvaluator'],  #same as name
   install_requires=reqs
)