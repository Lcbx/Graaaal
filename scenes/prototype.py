"""Combat prototype: actor panels and health bars; combat actions remain a placeholder.

Run with `python main.py scenes/prototype.py`. ImGui shows actor names and health;
the player panel is blue and enemy panels are red. Model helpers can be
imported without initializing a graphics device.
"""

from dataclasses import dataclass
from enum import IntFlag, auto

from numpyecs.ECS import ECS, Entity, component


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


@dataclass(frozen=True)
class ActorStatus:
	name: str
	health: int
	max_health: int
	is_player: bool

	@property
	def health_fraction(self):
		return min(1.0, max(0.0, self.health/self.max_health)) if self.max_health > 0 else 0.0


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


def actor_statuses(world, query):
	"""Read aligned actor fields without any graphics dependencies."""
	matched = query.fetch()
	health = matched[Damageable].selection.health
	attributes = matched[Attributes].selection
	flags = matched[ActorMetadata].selection.value
	return tuple(ActorStatus(name, int(current), int(maximum), bool(flag & ActorMetadataEnum.isPlayer))
		for name, current, maximum, flag in zip(attributes.name, health, attributes.max_health, flags))


class ActorCache:
	"""Refresh displayed data only when actor membership or fields change."""
	def __init__(self, world, query):
		self.world, self.query = world, query
		self.health = world.get(Damageable)
		self.attributes = world.get(Attributes)
		self.metadata = world.get(ActorMetadata)
		self.actors = ()
		self._version = None

	def update(self):
		version = (self.query.fetch(), self.health.version("health"),
			self.attributes.version("name", "max_health"), self.metadata.version("value"))
		if version == self._version:
			return False
		self.actors = actor_statuses(self.world, self.query)
		self._version = version
		return True


def draw_actors(cache):
	from imgui_bundle import imgui

	imgui.set_next_window_pos((30, 100), imgui.Cond_.first_use_ever)
	imgui.set_next_window_size((1100, 320), imgui.Cond_.first_use_ever)
	expanded, _ = imgui.begin("Combat prototype")
	try:
		if not expanded:
			return
		imgui.text("Actors")
		if cache.actors and imgui.begin_table("actors", len(cache.actors), imgui.TableFlags_.sizing_stretch_same):
			try:
				for index, actor in enumerate(cache.actors):
					imgui.table_next_column()
					imgui.push_id(index)
					color = (0.12, 0.25, 0.45, 1) if actor.is_player else (0.38, 0.14, 0.13, 1)
					imgui.push_style_color(imgui.Col_.child_bg, color)
					imgui.begin_child("actor", (0, 160))
					try:
						imgui.text(actor.name)
						imgui.text("Player" if actor.is_player else "Enemy")
						imgui.spacing()
						fill = (0.15, 0.75, 0.3, 1) if actor.health_fraction > 0.3 else (0.85, 0.2, 0.1, 1)
						imgui.push_style_color(imgui.Col_.plot_histogram, fill)
						imgui.progress_bar(actor.health_fraction, (-1, 24), f"{actor.health}/{actor.max_health}")
						imgui.pop_style_color()
					finally:
						imgui.end_child()
						imgui.pop_style_color()
						imgui.pop_id()
			finally:
				imgui.end_table()
		elif not cache.actors:
			imgui.text("No actors")
		imgui.text_disabled("Combat action selection and resolution are not implemented yet.")
	finally:
		imgui.end()


def run_scene():
	from numpyecs.ImGui import ImGui
	from numpyecs.RenderContext import RenderContext

	world = create_world()
	actors = world.query(Damageable, Attributes, ActorMetadata)
	idle_enemies = world.query(ActorMetadata, exclude=Action)
	cache = ActorCache(world, actors)
	RenderContext.init_window(1200, 650, "Combat prototype", target_fps=60)
	with ImGui() as gui:
		while RenderContext.window_loop():
			# TODO: choose and resolve actions for these entities using combat rules.
			enemies = inactive_enemy_ids(world, idle_enemies)
			cache.update()
			gui.render(lambda: draw_actors(cache), clear=(0.035, 0.04, 0.055, 1))


if __name__ == "__main__":
	run_scene()
