from random import randint
from ursina import *


app = Ursina()

window.size = (800,800)
window.color = color.rgb(102/255, 51/255, 0)
camera.orthographic = True
camera.fov = 1
positions = [(0,7),(7,7),(7,1),(3,2),(5,4),(2,6)]
objectsToBeDrawn = ['leaf', 'rock', 'sticks', 'trash', 'leaf', 'leaf']
objects = []

antStart = (0, 7)

while antStart in positions:
    antStart = (randint(0,7),randint(0,7))
    
random.shuffle(objectsToBeDrawn)
for i in range(6):
    name = objectsToBeDrawn[i]
    texturePath = 'images/' + name + '.png'

    objects.append(Entity(scale=(1/15, 1/15), model='quad', texture=texturePath, origin_x=0, origin_y=0,x=-.438 + 1/8*positions[i][1], y=.438 - 1/8*positions[i][0], z=-1, collider = 'box', name = name))

for i in range(8):
    for j in range(8):
        texture = 'images/ground.jpg'

        if (i,j) == antStart:
            texture = 'images/anthill.jpg'	

        Entity(scale=(1/8, 1/8), model='quad', texture=texture, origin_x=0, origin_y=0, x=-.438 + 1/8*j, y=.438 - 1/8*i, z = 0)

ant = Entity(scale=(1/10, 1/10), model='quad', texture='images/ant.png', origin_x=0, origin_y=0,x=-.438 + 1/8 * antStart[1], y=.438 - 1/8 * antStart[0] , z=-1, collider = 'box', keydown_cooldown = .15, carrying = False, collision_cooldown = .15, carried_obj= None, relative_coordinate = antStart, desiredObjectsCollected = 0)

cooldown = .15

def update():
    ant.keydown_cooldown -= time.dt
    ant.collision_cooldown -= time.dt

    if ant.keydown_cooldown > 0:
        return

    direction_x = held_keys['d'] - held_keys['a']
    direction_y = held_keys['w'] - held_keys['s']

    x = ant.x + direction_x * 1/8
    y = ant.y + direction_y * 1/8
    
    if direction_x != 0 or direction_y != 0:
        if x >= -.438 and x <= .438 and y >= -.438 and y <= .438:
            ant.relative_coordinate = (ant.relative_coordinate[0] + direction_y, ant.relative_coordinate[1] + direction_x)
            if direction_x != 0:
                ant.x = x
                ant.rotation_z = 90 * direction_x
            elif direction_y != 0:
                ant.y = y
                ant.rotation_z = 180 * (direction_y == -1)
    
        ant.keydown_cooldown = cooldown

    if ant.collision_cooldown > 0:
        return

    hit_info = ant.intersects()

    if hit_info.hit:
        ant.collision_cooldown = cooldown
        if hit_info.entity.name in ['leaf', 'rock', 'sticks', 'trash'] and not ant.carrying and held_keys['space']:
            ant.carrying = True
            ant.carried_obj = hit_info.entity
            hit_info.entity.disable()
            ant.texture = 'images/antwith' + hit_info.entity.name + '.png'
    elif ant.carrying and held_keys['space'] and ant.relative_coordinate == antStart:
        ant.collision_cooldown = cooldown
        ant.carrying = False
        ant.texture = 'images/ant.png'
        ant.carried_obj.x = ant.x
        ant.carried_obj.y = ant.y
        
        if ant.carried_obj.name in ['leaf', 'sticks']:
            ant.desiredObjectsCollected += 1
            if ant.desiredObjectsCollected == 4:
                print("You win!")
                quit()
        objects.remove(ant.carried_obj)
        destroy(ant.carried_obj)
        ant.carried_obj = None
        
app.run()