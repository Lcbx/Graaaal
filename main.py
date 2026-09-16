#!python

if __name__ == "__main__":
	import sys
	from argparse import ArgumentParser
	from pathlib import Path
	from importlib import import_module

	import src

	parser = ArgumentParser(
		prog='main.py',
		description='Graaaal, game engine on top of numpy and wgpu',
		epilog='link: https://github.com/Lcbx/Graaaal')
	parser.add_argument('scene', nargs='?', help='set scene script (to run or compile)')
	parser.add_argument('-t', '--test', nargs='?', const=True, default=False, help='launches unit tests or a single test if specified by name')
	parser.add_argument('-c', '--compile', action='store_true', help='compile scene into standalone executable')
	args = parser.parse_args()

	normalize_path = lambda p: Path(p).as_posix()
	to_module = lambda p: normalize_path(p).replace('/', '.').replace('.py', '')

	if not ( args.scene or args.test or args.compile):
		parser.error("scene is required unless --test or --compile are used")

	# run test suite using pytest
	elif args.test:
		import pytest
		from glob import glob
		test_paths = glob('./tests/*.py')
		pytest_args = test_paths


		# run particular test
		if type(args.test) is str and (
			test_name := args.test.split('::')[-1] ).startswith('test'):
			module_path = [ tp for tp in test_paths
				if hasattr((mod := import_module( to_module(tp) )), test_name)
			]

			if not module_path:
				print(f'could not find "{args.test}" test. candidates:')
				from pprint import pp 
				pp( list(f for m in module_path for f in m.__dict__ ))
				exit(1)

			pytest_args = [ f'{module_path[0]}::{test_name}']
		ret = pytest.main(['-qs', *pytest_args] )
		exit(ret)
			

	# run or compile the scene
	else:
		scene = args.scene
		if args.compile:
			from subprocess import run
			cwd = src.__path__[0]
			run( f'py -m mypyc -m ECS', cwd=cwd)
			run( f'py -m mypyc -m RenderContext', cwd=cwd)
			run( f'py -m mypyc -m Utils', cwd=cwd)
			run( f'py -m mypyc -m DrawBatches', cwd=cwd)
			if scene: run( f'py -m nuitka --output-dir=build --standalone {scene}'.split(' ') )
		else:
			import_module( to_module(scene) )
