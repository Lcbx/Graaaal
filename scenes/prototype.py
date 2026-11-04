from ecs import *
import shader_util as su

@component
class Damageable:
	health: int

class ActorMetadataEnum(IntFlag):
    isPlayer=auto()
    lowHealth=auto()
    bleeding=auto()
    poisoned=auto()

@component
class ActorMetadata:
	value : ActorMetadataEnum

@component
class Attributes:
	name: str

	max_health:int

	damage:float
	speed:float
	defense:float

@component
class Essences:
	essence0:str
	essence1:str
	essence2:str
	essence3:str
	essence4:str

@component
class Action:
	currentAction:str
	started:float
	end:float
	target:Entity


world = ECS()
world.register(Damageable, Attributes, ActorMetadata, Essences, Action)

playerEntity = world.create_entity(
	ActorMetadata(ActorMetadataEnum.isPlayer),
	Damageable(150),
	Attributes('player', 100, 1.0,1.0,1.0),
	Essences('primitive', None, None, None, None))

world.create_entity(
	ActorMetadata(0),
	Damageable(30),
	Attributes('mosquito', 30, 0.5,1.2,1.0),
	Essences('bloodSucker', None, None, None, None))
world.create_entity(
	ActorMetadata(0),
	Damageable(50),
	Attributes('goblin', 50, 0.8,1.0,1.0),
	Essences('primitive', None, None, None, None))



WINDOW_w, WINDOW_h = 1800, 900
su.InitWindow(WINDOW_w, WINDOW_h, "combat system prototype")
su.rl.SetTargetFPS(60)

actionsStore = world.get_store(Action)
attributesStore = world.get_store(Attributes)
metadataStore = world.get_store(ActorMetadata)

while not su.rl.WindowShouldClose():
	su.rl.BeginDrawing()
	su.ClearBuffers()

	time = su.rl.GetTime()

	entitiesActive = world.where(Action)
	enemies = metadataStore.query(lambda value : value & ActorMetadataEnum.isPlayer == 0)
	#print(enemies)

	#for entity in enemies:
	#	print(attributesStore.get(entity))

	enemiesInactive = np.setdiff1d(enemies, entitiesActive)
	#print(enemiesInactive)
	for entity in enemiesInactive:
		# TODO : add an action based on their essence
		pass

	#TODO resolve and apply actions

	for i in range(1,6):
		anchor = (WINDOW_w/6*i, WINDOW_h/3)
		w, h = WINDOW_w/8, WINDOW_h/3
		x = anchor[0]-w/2
		y = anchor[1]-h/2
		rect = (x, y, w, h)
		su.rl.GuiGroupBox( rect, f"enemy{i}".encode() )
		su.rl.DrawRectangleRec( (x, y+h, w, 20), su.rl.RED)
		su.rl.DrawRectangleRec( (x, y+h+20, w, 20), su.rl.YELLOW)


	su.rl.EndDrawing()