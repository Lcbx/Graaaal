#!python

if __name__ == "__main__":
	import sys
	from argparse import ArgumentParser
	from pathlib import Path
	from importlib import import_module

	import numpyecs

	parser = ArgumentParser(
		prog='main.py',
		description='Graaaal, game engine on top of numpy and wgpu',
		epilog='link: https://github.com/Lcbx/Graaaal')
	parser.add_argument('scene', nargs='?', help='set scene script (to run or compile)')
	parser.add_argument('-t', '--test', nargs='?', const=True, default=False, help='launches unit tests or a single test if specified by name')
	parser.add_argument('-c', '--compile', action='store_true', help='compile scene into standalone executable')
	parser.add_argument('--frames', type=int, help='stop a RenderContext scene after this many frames (runtime smoke check)')
	args = parser.parse_args()
	if args.frames is not None and (args.frames <= 0 or not args.scene or args.test or args.compile):
		parser.error('--frames requires a scene, a positive frame count, and normal run mode')

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
			cwd = numpyecs.__path__[0]
			run( f'py -m mypyc -m numpyecs.ECS', cwd=cwd)
			run( f'py -m mypyc -m numpyecs.RenderContext', cwd=cwd)
			run( f'py -m mypyc -m numpyecs.Math', cwd=cwd)
			run( f'py -m mypyc -m numpyecs.DrawBatches', cwd=cwd)
			if scene: run( f'py -m nuitka --output-dir=build --standalone {scene}'.split(' ') )
		else:
			from numpyecs.RenderContext import RenderContext
			if args.frames is not None:
				window_loop = RenderContext.window_loop
				frames = 0
				def bounded_window_loop():
					global frames
					# Present the final submitted image before ending the scene.
					if not window_loop() or frames >= args.frames:
						return False
					frames += 1
					return True
				RenderContext.window_loop = bounded_window_loop
			try:
				module = import_module(to_module(scene))
				# New scenes can expose an entry point without opening a window on import.
				if callable(getattr(module, "run_scene", None)):
					module.run_scene()
			finally:
				if args.frames is not None:
					RenderContext.window_loop = window_loop
				getattr(RenderContext, "cleanup", lambda: None)()
			if args.frames is not None:
				print(f"Scene completed: {frames} frames")
