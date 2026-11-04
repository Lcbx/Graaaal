"""Combat prototype: actor panels and health bars; combat actions remain a placeholder.

Run with `python main.py scenes/prototype.py`. Names/health appear in the window
title; the player panel is blue and enemy panels are red. Model helpers can be
imported without initializing a graphics device.
"""
from enum import IntFlag, auto

import numpy as np

from ECS import ECS, Entity, component


@component
class Damageable:
	health: int


class ActorMetadataEnum(IntFlag):
	isPlayer = auto()
	lowHealth = auto()
	bleeding = auto()
	poisoned = auto()


@component
class ActorMetadata:
	value: ActorMetadataEnum


@component
class Attributes:
	name: str
	max_health: int
	damage: float
	speed: float
	defense: float


@component
class Essences:
	essence0: str
	essence1: str
	essence2: str
	essence3: str
	essence4: str


@component
class Action:
	currentAction: str
	started: float
	end: float
	target: Entity


PANEL_DTYPE = np.dtype([("iRect", np.float32, 4), ("iColor", np.float32, 4)])


def create_world():
	world = ECS()
	world.register(Damageable, Attributes, ActorMetadata, Essences, Action)
	for flags, health, attributes, essence in (
		(ActorMetadataEnum.isPlayer, 150, Attributes("player", 100, 1, 1, 1), "primitive"),
		(ActorMetadataEnum(0), 30, Attributes("mosquito", 30, 0.5, 1.2, 1), "bloodSucker"),
		(ActorMetadataEnum(0), 50, Attributes("goblin", 50, 0.8, 1, 1), "primitive"),
	):
		world.add(world.create(), ActorMetadata(flags), Damageable(health), attributes,
			Essences(essence, None, None, None, None))
	return world


def inactive_enemy_ids(world, query):
	metadata = world.get(ActorMetadata)
	rows = metadata.query_rows(lambda value: (value & ActorMetadataEnum.isPlayer) == 0, query.entities())
	return metadata.owner_ids(rows)


def actor_panels(world, query):
	"""Build normalized screen rectangles from aligned actor components."""
	matched = query.fetch()
	health = matched[Damageable].selection.health
	attributes = matched[Attributes].selection
	flags = matched[ActorMetadata].selection.value
	maximum = attributes.max_health
	ratios = np.divide(health, maximum, out=np.zeros(len(health)), where=maximum > 0)
	ratios = np.clip(ratios, 0, 1)
	panels = np.zeros(3 * len(matched), dtype=PANEL_DTYPE)
	for i, ratio in enumerate(ratios):
		x, width = 0.05 + i * 0.9 / len(matched), 0.8 / len(matched)
		panel, track, fill = panels[3 * i:3 * i + 3]
		panel["iRect"] = (x, 0.3, width, 0.32)
		panel["iColor"] = (0.12, 0.25, 0.45, 1) if flags[i] & ActorMetadataEnum.isPlayer else (0.38, 0.14, 0.13, 1)
		track["iRect"] = (x, 0.65, width, 0.045)
		track["iColor"] = (0.12, 0.12, 0.14, 1)
		fill["iRect"] = (x, 0.65, width * ratio, 0.045)
		fill["iColor"] = (0.15, 0.75, 0.3, 1) if ratio > 0.3 else (0.85, 0.2, 0.1, 1)
	title = "Combat prototype | " + " | ".join(
		f"{name}: {current}/{limit}" for name, current, limit in zip(attributes.name, health, maximum))
	return panels, title


class PanelCache:
	"""Rebuild panels only when actor membership or displayed fields change."""
	def __init__(self, world, query):
		self.world, self.query = world, query
		self.health = world.get(Damageable)
		self.attributes = world.get(Attributes)
		self.metadata = world.get(ActorMetadata)
		self.panels = np.empty(0, dtype=PANEL_DTYPE)
		self.title = ""
		self._version = None

	def update(self):
		version = (self.query.fetch(), self.health.version("health"),
			self.attributes.version("name", "max_health"), self.metadata.version("value"))
		if version == self._version:
			return False
		self.panels, self.title = actor_panels(self.world, self.query)
		self._version = version
		return True


def run_scene():
	# Keep graphics imports/device setup out of the CPU model helpers.
	from pathlib import Path
	import glfw
	from RenderContext import RenderContext, Shader, RenderPipeline, GpuBuffer, BufferUsage

	world = create_world()
	actors = world.query(Damageable, Attributes, ActorMetadata)
	idle_enemies = world.query(ActorMetadata, exclude=Action)
	cache = PanelCache(world, actors)
	cache.update()
	panels, title = cache.panels, cache.title
	RenderContext.init_window(1200, 650, title, target_fps=60)
	shader = Shader(filepath=str(Path(__file__).with_name("shaders") / "prototype.shader"))
	pipeline = RenderPipeline(shader, vertex_entry="vertex", fragment_entry="fragment", depth_test=None, depth_write=False, cull_mode="none")
	buffer = GpuBuffer(panels, BufferUsage.VERTEX | BufferUsage.COPY_DST)
	previous_title = title
	while RenderContext.window_loop():
		# TODO: choose and resolve actions for these entities using combat rules.
		enemies = inactive_enemy_ids(world, idle_enemies)
		if cache.update():
			panels, title = cache.panels, cache.title
			if title != previous_title:
				glfw.set_window_title(RenderContext.window, title)
				previous_title = title
			if len(panels) != len(buffer.content):
				buffer.resize(len(panels))
			buffer.content = panels
			if len(panels):
				buffer.upload()
		width, height = RenderContext.windowDimensions
		if width <= 0 or height <= 0:
			continue
		cmd = RenderContext.commands("prototype")
		with cmd.render_pass(color=RenderContext.screen(clear=(0.035, 0.04, 0.055, 1))) as rp:
			rp.set_pipeline(pipeline)
			if len(panels):
				rp.set_vertex_buffer(0, buffer, step_mode="instance")
				rp.draw(6, instance_count=len(panels))
		RenderContext.submit(cmd.finish())


if __name__ == "__main__":
	run_scene()
