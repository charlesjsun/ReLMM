import numpy as np

from .utils import *
from .objects import StaticObject, TexturedBox, Wall, FloorPatch

from softlearning.environments.helpers import random_point_in_circle
import pybullet_data
OSS_DATA_ROOT = pybullet_data.getDataPath()

def initialize_room(locobot_interface, name, room_params={}):
    if name == "simple":
        return SimpleRoom(locobot_interface, room_params)
    elif name == "simple_texture":
        return SimpleTextureRoom(locobot_interface, room_params)
    elif name == "simple_obstacles":
        return SimpleRoomWithObstacles(locobot_interface, room_params)
    elif name == "medium":
        return MediumRoom(locobot_interface, room_params)
    elif name == "grasping":
        return GraspingRoom(locobot_interface, room_params)
    elif name == "grasping_2":
        return GraspingRoom2(locobot_interface, room_params)
    elif name == "single":
        return SingleRoom(locobot_interface, room_params)
    elif name == "double":
        return DoubleRoom(locobot_interface, room_params)
    elif name == "double_v2":
        return DoubleV2Room(locobot_interface, room_params)
    else:
        return NotImplementedError(f"no room has name {name}")

class Room:
    def __init__(self, interface, params):
        self.interface = interface
        self.objects_id = []
        self.obstacles_id = []
        self.params = params

    def reset(self, *args, **kwargs):
        return NotImplementedError

    @property
    def num_objects(self):
        return len(self.objects_id)

    @property
    def extent(self):
        """ Furthest distance from the origin considered to be in the room. """
        return 100

    @property
    def object_discard_pos(self):
        return [self.extent * 3, 0, 1]

class SimpleRoom(Room):
    """ Simple room that has a wall and objects inside. """
    def __init__(self, interface, params):
        defaults = dict(
            num_objects=100, 
            object_name="greensquareball", 
            wall_size=5.0,
            no_spawn_radius=1.0,
            wall_urdf="walls",
        )
        defaults.update(params)
        super().__init__(interface, defaults)

        self._wall_size = self.params["wall_size"]
        self.obstacles_id.append(self.interface.spawn_object(URDF[self.params["wall_urdf"]], scale=self._wall_size, pos=[0, 0, -1.0]))

        self._no_spawn_radius = self.params["no_spawn_radius"]
        
        self._num_objects = self.params["num_objects"]
        for i in range(self._num_objects):
            self.objects_id.append(self.interface.spawn_object(URDF[self.params["object_name"]], np.array([0.0, 0.0, 10 + i])))

    def is_valid_spawn_loc(self, x, y, robot_pos=[0, 0]):
        return not is_in_circle(x, y, robot_pos[0], robot_pos[1], self._no_spawn_radius)

    def reset(self):
        for i in range(self._num_objects):
            for _ in range(5000):
                x, y = np.random.uniform(-self._wall_size * 0.5, self._wall_size * 0.5, size=(2,))
                if self.is_valid_spawn_loc(x, y):
                    break
            self.interface.move_object(self.objects_id[i], [x, y, 0.015])
    
    @property
    def num_objects(self):
        return self._num_objects

    @property
    def extent(self):
        return self._wall_size * 2.0

class SimpleTextureRoom(SimpleRoom):
    """ Simple room that has textured floors and walls. """
    def __init__(self, interface, params):
        defaults = dict(
            wall_urdf="walls_2",
        )
        defaults.update(params)
        super().__init__(interface, defaults)

        self.interface.load_floor(URDF["floor"], globalScaling=self._wall_size * 0.5)
        self.interface.change_floor_texture("floor2")

        # object discard box
        self.obstacles_id.append(self.interface.spawn_object(URDF[self.params["wall_urdf"]], scale=2.0, pos=[self._wall_size * 3, 0, 0]))
        self.obstacles_id.append(self.interface.spawn_object(URDF["floor"], scale=1.0, pos=[self._wall_size * 3, 0, 0]))

    @property
    def object_discard_pos(self):
        return [self._wall_size * 3 + np.random.uniform(-0.9, 0.9), np.random.uniform(-0.9, 0.9), np.random.uniform(0.1, 0.5)]

    def random_robot_pos_yaw(self):
        robot_pos = np.random.uniform(-1.9, 1.9, size=(2,))
        robot_yaw = np.random.uniform(0, np.pi * 2)
        return robot_pos, robot_yaw





class BaseTileRoom(Room):
    def __init__(self, interface, params):
        defaults = dict(
            num_objects=50,
            object_name="greensquareball",
            robot_pos=[0, 0],
        )
        defaults.update(params)
        super().__init__(interface, defaults)

        self.interface.remove_object(self.interface.plane_id)

        self.floors = []
        self.generate_floors()

        self.walls = []
        self.generate_walls()

        self.discard_walls = []
        self.discard_floor = None
        self.generate_discard_box()

        # objects
        self._num_objects = self.params["num_objects"]
        object_name = self.params["object_name"]
        if isinstance(object_name, (list, tuple)):
            num_per_type = self._num_objects // len(object_name)
            for name in object_name:
                for _ in range(num_per_type):
                    self.objects_id.append(self.interface.spawn_object(URDF[name], self.object_discard_pos))
                    print(name)
            while len(self.objects_id) < self._num_objects:
                self.objects_id.append(self.interface.spawn_object(URDF[name], self.object_discard_pos))
        else:
            for _ in range(self._num_objects):
                self.objects_id.append(self.interface.spawn_object(URDF[object_name], self.object_discard_pos))
                print(object_name)

        self.robot_pos = self.params["robot_pos"]

    def generate_floors(self):
        raise NotImplementedError

    def generate_walls(self):
        raise NotImplementedError

    def generate_discard_box(self):
        self.discard_walls.append(Wall(self.interface, [10.5, 0, 0], [-1.0, 0.0], 1.0))
        self.discard_walls.append(Wall(self.interface, [9.5, 0, 0], [1.0, 0.0], 1.0))
        self.discard_walls.append(Wall(self.interface, [10, 0.5, 0], [0.0, -1.0], 1.0))
        self.discard_walls.append(Wall(self.interface, [10, -0.5, 0], [0.0, 1.0], 1.0))
        self.discard_floor = FloorPatch(self.interface, [10.0, 0.0, 0.0], "floor_wood_3", 1.0)

    @property
    def object_discard_pos(self):
        return [10 + np.random.uniform(-0.45, 0.45), np.random.uniform(-0.45, 0.45), np.random.uniform(0.1, 0.5)]

    @property
    def num_objects(self):
        return self._num_objects

    def reset_object(self, object_ind, robot_pos, max_radius=np.inf):
        for _ in range(5000):
            floor = np.random.choice(self.floors)
            x, y = floor.get_random_loc()
            if not is_in_circle(x, y, robot_pos[0], robot_pos[1], 0.5) and is_in_circle(x, y, robot_pos[0], robot_pos[1], max_radius):
                break
        self.interface.move_object(self.objects_id[object_ind], [x, y, 0.015])

    def reset(self):
        for i in range(self._num_objects):
            for _ in range(5000):
                floor = np.random.choice(self.floors)
                x, y = floor.get_random_loc()
                if not is_in_circle(x, y, self.robot_pos[0], self.robot_pos[1], 0.5):
                    break
            self.interface.move_object(self.objects_id[i], [x, y, 0.015])

    def is_object_in_bound(self, object_ind):
        object_pos, _ = self.interface.get_object(self.objects_id[object_ind])
        for floor in self.floors:
            if floor.is_in_bound(object_pos[0], object_pos[1]):
                return True
        return False

    def is_object_in_discard(self, object_ind):
        object_pos, _ = self.interface.get_object(self.objects_id[object_ind])
        return self.discard_floor.is_in_bound(object_pos[0], object_pos[1])
    
    def force_object_in_bound(self, object_ind):
        object_pos, _ = self.interface.get_object(self.objects_id[object_ind])
        closest_floor = min(self.floors, key=lambda f: (f.pos[0] - object_pos[0]) ** 2 + (f.pos[1] - object_pos[1]) ** 2)
        dx = object_pos[0] - closest_floor.pos[0]
        dy = object_pos[1] - closest_floor.pos[1]
        max_norm = max(abs(dx), abs(dy)) * 2.0
        x = closest_floor.pos[0] + (dx / max_norm) * 0.85
        y = closest_floor.pos[1] + (dy / max_norm) * 0.85
        self.interface.move_object(self.objects_id[object_ind], [x, y, 0.015])

    def force_object_in_bound_if_not(self, object_ind):
        if not self.is_object_in_discard(object_ind) and not self.is_object_in_bound(object_ind):
            self.force_object_in_bound(object_ind)

    def get_turn_direction_if_should_turn(self):
        x, y, yaw = self.interface.get_base_pos_and_yaw()
        walls = []
        for w in self.walls:
            if w.in_turn_detection_box(x, y):
                walls.append(w) 

        if len(walls) == 0:
            return None

        robot_facing = np.array([np.cos(yaw), np.sin(yaw)])
        wall_normal_x = sum(w.facing[0] for w in walls)
        wall_normal_y = sum(w.facing[1] for w in walls)
        wall_normal = np.array([wall_normal_x, wall_normal_y]) / np.sqrt(wall_normal_x ** 2 + wall_normal_y ** 2)

        dot_prod = robot_facing.dot(wall_normal)
        if dot_prod >= 0.0:
            return None

        direction = wall_normal[0] * (-robot_facing[1]) + wall_normal[1] * robot_facing[0]
        if direction <= 0:
            return "right", dot_prod
        else:
            return "left", dot_prod

    def is_in_turn_detection_box(self, x, y):
        for w in self.walls:
            if w.in_turn_detection_box(x, y):
                return True
        return False

    def random_robot_pos_yaw(self):
        for _ in range(5000):
            floor = np.random.choice(self.floors)
            x, y = floor.get_random_loc()
            if not self.is_in_turn_detection_box(x, y):
                break
        yaw = np.random.uniform(0, np.pi * 2.0)
        return np.array([x, y]), yaw


class SingleRoom(BaseTileRoom):

    def __init__(self, interface, params):
        defaults = dict(
            single_floor=True,
            single_floor_texture="floor_wood_2",
        )
        defaults.update(params)
        super().__init__(interface, defaults)
    
    def generate_floors(self):
        # starting room
        if self.params["single_floor"]:
            tex = self.params["single_floor_texture"]
            self.floors.append(FloorPatch(self.interface, [-1.0,-1.0, 0.0], tex, 1.0, neg_x_wall=True, neg_y_wall=True))
            self.floors.append(FloorPatch(self.interface, [ 0.0,-1.0, 0.0], tex, 1.0,                  neg_y_wall=True))
            self.floors.append(FloorPatch(self.interface, [ 1.0,-1.0, 0.0], tex, 1.0, pos_x_wall=True, neg_y_wall=True))
            
            self.floors.append(FloorPatch(self.interface, [-1.0, 0.0, 0.0], tex, 1.0, neg_x_wall=True))
            self.floors.append(FloorPatch(self.interface, [ 0.0, 0.0, 0.0], tex, 1.0))
            self.floors.append(FloorPatch(self.interface, [ 1.0, 0.0, 0.0], tex, 1.0, pos_x_wall=True))
            
            self.floors.append(FloorPatch(self.interface, [-1.0, 1.0, 0.0], tex, 1.0, neg_x_wall=True, pos_y_wall=True))
            self.floors.append(FloorPatch(self.interface, [ 0.0, 1.0, 0.0], tex, 1.0,                  pos_y_wall=True))
            self.floors.append(FloorPatch(self.interface, [ 1.0, 1.0, 0.0], tex, 1.0, pos_x_wall=True, pos_y_wall=True))

        else:
            self.floors.append(FloorPatch(self.interface, [-1.0,-1.0, 0.0], "floor_wood_3",   1.0, neg_x_wall=True, neg_y_wall=True))
            self.floors.append(FloorPatch(self.interface, [ 0.0,-1.0, 0.0], "floor_wood_2",   1.0,                  neg_y_wall=True))
            self.floors.append(FloorPatch(self.interface, [ 1.0,-1.0, 0.0], "floor_marble_2", 1.0, pos_x_wall=True, neg_y_wall=True))
            
            self.floors.append(FloorPatch(self.interface, [-1.0, 0.0, 0.0], "floor_carpet_5", 1.0, neg_x_wall=True))
            self.floors.append(FloorPatch(self.interface, [ 0.0, 0.0, 0.0], "floor_wood_1",   1.0))
            self.floors.append(FloorPatch(self.interface, [ 1.0, 0.0, 0.0], "floor_carpet_4", 1.0, pos_x_wall=True))
            
            self.floors.append(FloorPatch(self.interface, [-1.0, 1.0, 0.0], "floor_marble_1", 1.0, neg_x_wall=True, pos_y_wall=True))
            self.floors.append(FloorPatch(self.interface, [ 0.0, 1.0, 0.0], "floor_carpet_7", 1.0,                  pos_y_wall=True))
            self.floors.append(FloorPatch(self.interface, [ 1.0, 1.0, 0.0], "floor_marble_3", 1.0, pos_x_wall=True, pos_y_wall=True))
            
    def generate_walls(self):
        # -x side
        self.walls.append(Wall(self.interface, [-1.5,-1.0, 0], [ 1, 0], 1.0))
        self.walls.append(Wall(self.interface, [-1.5, 0.0, 0], [ 1, 0], 1.0))
        self.walls.append(Wall(self.interface, [-1.5, 1.0, 0], [ 1, 0], 1.0))

        # +x side
        self.walls.append(Wall(self.interface, [ 1.5,-1.0, 0], [-1, 0], 1.0))
        self.walls.append(Wall(self.interface, [ 1.5, 0.0, 0], [-1, 0], 1.0))
        self.walls.append(Wall(self.interface, [ 1.5, 1.0, 0], [-1, 0], 1.0))

        # -y side
        self.walls.append(Wall(self.interface, [-1.0,-1.5, 0], [ 0, 1], 1.0))
        self.walls.append(Wall(self.interface, [ 0.0,-1.5, 0], [ 0, 1], 1.0))
        self.walls.append(Wall(self.interface, [ 1.0,-1.5, 0], [ 0, 1], 1.0))

        # +y side
        self.walls.append(Wall(self.interface, [-1.0, 1.5, 0], [ 0,-1], 1.0))
        self.walls.append(Wall(self.interface, [ 0.0, 1.5, 0], [ 0,-1], 1.0))
        self.walls.append(Wall(self.interface, [ 1.0, 1.5, 0], [ 0,-1], 1.0))

    def random_robot_pos_yaw(self):
        x = np.random.uniform(-0.75, 0.75)
        y = np.random.uniform(-0.75, 0.75)
        yaw = np.random.uniform(0, np.pi * 2.0)
        return np.array([x, y]), yaw


class DoubleV2Room(BaseTileRoom):
    def __init__(self, interface, params):
        defaults = dict(
            floor_texture_1="floor_wood_2",
            floor_texture_2="floor_wood_3",
        )
        defaults.update(params)
        super().__init__(interface, defaults)

    def generate_floors(self):
        # starting room
        floor_texture_1 = self.params["floor_texture_1"]
        self.floors.append(FloorPatch(self.interface, [ 0.0, 0.0, 0.0], floor_texture_1, 1.0, neg_y_wall=True))
        self.floors.append(FloorPatch(self.interface, [ 1.0, 0.0, 0.0], floor_texture_1, 1.0, neg_y_wall=True, pos_y_wall=True))
        self.floors.append(FloorPatch(self.interface, [-1.0, 0.0, 0.0], floor_texture_1, 1.0, neg_y_wall=True, neg_x_wall=True))
        self.floors.append(FloorPatch(self.interface, [ 0.0, 1.0, 0.0], floor_texture_1, 1.0))
        self.floors.append(FloorPatch(self.interface, [ 1.0, 1.0, 0.0], floor_texture_1, 1.0, pos_y_wall=True, pos_x_wall=True))
        self.floors.append(FloorPatch(self.interface, [-1.0, 1.0, 0.0], floor_texture_1, 1.0, pos_y_wall=True, neg_x_wall=True))

        # other room
        floor_texture_2 = self.params["floor_texture_2"]
        self.floors.append(FloorPatch(self.interface, [ 0.0, 2.0, 0.0], floor_texture_2, 1.0))
        self.floors.append(FloorPatch(self.interface, [ 1.0, 2.0, 0.0], floor_texture_2, 1.0, neg_y_wall=True, pos_x_wall=True))
        self.floors.append(FloorPatch(self.interface, [-1.0, 2.0, 0.0], floor_texture_2, 1.0, neg_y_wall=True, neg_x_wall=True))
        self.floors.append(FloorPatch(self.interface, [ 0.0, 3.0, 0.0], floor_texture_2, 1.0, pos_y_wall=True))
        self.floors.append(FloorPatch(self.interface, [ 1.0, 3.0, 0.0], floor_texture_2, 1.0, pos_y_wall=True, pos_x_wall=True))
        self.floors.append(FloorPatch(self.interface, [-1.0, 3.0, 0.0], floor_texture_2, 1.0, pos_y_wall=True, neg_x_wall=True))

    def generate_walls(self):
        # -y side room 1
        self.walls.append(Wall(self.interface, [-1.0, -0.5, 0], [ 0,  1], 1.0))
        self.walls.append(Wall(self.interface, [ 0.0, -0.5, 0], [ 0,  1], 1.0))
        self.walls.append(Wall(self.interface, [ 1.0, -0.5, 0], [ 0,  1], 1.0))
        # +x side room 1
        self.walls.append(Wall(self.interface, [ 1.5,  0.0, 0], [-1,  0], 1.0))
        self.walls.append(Wall(self.interface, [ 1.5,  1.0, 0], [-1,  0], 1.0))
        # -x side room 1
        self.walls.append(Wall(self.interface, [-1.5,  0.0, 0], [ 1,  0], 1.0))
        self.walls.append(Wall(self.interface, [-1.5,  1.0, 0], [ 1,  0], 1.0))
        # +y side room 1
        self.walls.append(Wall(self.interface, [ 1.0,  1.5, 0], [ 0, -1], 1.0, is_thin=True))
        self.walls.append(Wall(self.interface, [-1.0,  1.5, 0], [ 0, -1], 1.0, is_thin=True))

        # -y side room 2
        self.walls.append(Wall(self.interface, [ 1.0,  1.5, 0], [ 0,  1], 1.0, is_thin=True))
        self.walls.append(Wall(self.interface, [-1.0,  1.5, 0], [ 0,  1], 1.0, is_thin=True))
        # +x side room 2
        self.walls.append(Wall(self.interface, [ 1.5,  2.0, 0], [-1,  0], 1.0))
        self.walls.append(Wall(self.interface, [ 1.5,  3.0, 0], [-1,  0], 1.0))
        # -x side room 2
        self.walls.append(Wall(self.interface, [-1.5,  2.0, 0], [ 1,  0], 1.0))
        self.walls.append(Wall(self.interface, [-1.5,  3.0, 0], [ 1,  0], 1.0))
        # +y side room 2
        self.walls.append(Wall(self.interface, [-1.0,  3.5, 0], [ 0, -1], 1.0))
        self.walls.append(Wall(self.interface, [ 0.0,  3.5, 0], [ 0, -1], 1.0))
        self.walls.append(Wall(self.interface, [ 1.0,  3.5, 0], [ 0, -1], 1.0))


class DoubleRoom(BaseTileRoom):
    
    def generate_floors(self):
        # starting room
        self.floors.append(FloorPatch(self.interface, [0.0, 0.0, 0.0], "floor_wood_2", 1.0, neg_y_wall=True))
        self.floors.append(FloorPatch(self.interface, [1.0, 0.0, 0.0], "floor_marble_2", 1.0, neg_y_wall=True, pos_y_wall=True))
        self.floors.append(FloorPatch(self.interface, [-1.0, 0.0, 0.0], "floor_wood_2", 1.0, neg_y_wall=True, neg_x_wall=True))
        self.floors.append(FloorPatch(self.interface, [0.0, 1.0, 0.0], "floor_wood_1", 1.0))
        self.floors.append(FloorPatch(self.interface, [1.0, 1.0, 0.0], "floor_marble_2", 1.0, pos_y_wall=True, pos_x_wall=True))
        self.floors.append(FloorPatch(self.interface, [-1.0, 1.0, 0.0], "floor_carpet_5", 1.0, pos_y_wall=True, neg_x_wall=True))

        # corridor
        self.floors.append(FloorPatch(self.interface, [0.0, 2.0, 0.0], "floor_wood_3", 1.0, pos_x_wall=True, neg_x_wall=True))
        
        # other room
        self.floors.append(FloorPatch(self.interface, [0.0, 3.0, 0.0], "floor_carpet_3", 1.0))
        self.floors.append(FloorPatch(self.interface, [1.0, 3.0, 0.0], "floor_marble_3", 1.0, neg_y_wall=True, pos_x_wall=True))
        self.floors.append(FloorPatch(self.interface, [-1.0, 3.0, 0.0], "floor_marble_1", 1.0, neg_y_wall=True, neg_x_wall=True))
        self.floors.append(FloorPatch(self.interface, [0.0, 4.0, 0.0], "floor_carpet_1", 1.0, pos_y_wall=True))
        self.floors.append(FloorPatch(self.interface, [1.0, 4.0, 0.0], "floor_carpet_6", 1.0, pos_y_wall=True, pos_x_wall=True))
        self.floors.append(FloorPatch(self.interface, [-1.0, 4.0, 0.0], "floor_carpet_7", 1.0, pos_y_wall=True, neg_x_wall=True))

    def generate_walls(self):
        self.walls.append(Wall(self.interface, [1.5, 0.5, 0], [-1, 0], 2.0))
        self.walls.append(Wall(self.interface, [-1.5, 0.5, 0], [1, 0], 2.0))
        self.walls.append(Wall(self.interface, [0, -0.5, 0], [0, 1], 3.0))

        self.walls.append(Wall(self.interface, [1.0, 1.5, 0], [0, -1], 1.0))
        self.walls.append(Wall(self.interface, [-1.0, 1.5, 0], [0, -1], 1.0))

        self.walls.append(Wall(self.interface, [0.5, 2.0, 0], [-1, 0], 1.0))
        self.walls.append(Wall(self.interface, [-0.5, 2.0, 0], [1, 0], 1.0))

        self.walls.append(Wall(self.interface, [1.0, 2.5, 0], [0, 1], 1.0))
        self.walls.append(Wall(self.interface, [-1.0, 2.5, 0], [0, 1], 1.0))

        self.walls.append(Wall(self.interface, [1.5, 3.5, 0], [-1, 0], 2.0))
        self.walls.append(Wall(self.interface, [-1.5, 3.5, 0], [1, 0], 2.0))
        self.walls.append(Wall(self.interface, [0, 4.5, 0], [0, -1], 3.0))









class GraspingRoom(Room):
    """ Room with objects spawn around specific location. """
    def __init__(self, interface, params):
        defaults = dict(
            min_objects=1,
            max_objects=10,
            object_name="greensquareball", 
            spawn_loc = [0, 0.42],
            spawn_radius = 0.10,
            use_bin=False,
        )
        defaults.update(params)
        super().__init__(interface, defaults)
        
        self.use_bin = self.params['use_bin']
        self._spawn_loc = self.params["spawn_loc"]
        self._spawn_radius = self.params["spawn_radius"]
        #self.interface.change_floor_texture("wood")
        self.interface.load_floor(URDF["floor"], globalScaling=2.5)
        self.interface.change_floor_texture("bluerugs")
        #self.interface.change_floor_texture("floor2")
        self._min_objects = self.params["min_objects"]
        self._max_objects = self.params["max_objects"]
        for i in range(self._max_objects):
            print("object name", self.params["object_name"])
            self.objects_id.append(self.interface.spawn_object(URDF[self.params["object_name"]], np.array([0.0, 0.0, 10 + i])))
        if self.use_bin:
            import math
            self.tray_ori = self.interface.p.getQuaternionFromEuler((0,0,math.pi))
            self.tray_id = self.interface.p.loadURDF(
                OSS_DATA_ROOT+'/tray/tray.urdf', [.95, 0.5, 0.05], self.tray_ori, globalScaling=0.7,
                )
            self.spawn_radius = 0.05
        
    def is_valid_spawn_loc(self, x, y):
        return not is_in_circle(x, y, 0, 0, 0.2)

    def reset(self, num_objects=None):
        if not num_objects:
            num_objects = np.random.randint(self._min_objects, self._max_objects + 1)
        if self.use_bin:
            self.interface.move_object(self.tray_id, [0.45, 0, 0.0], self.tray_ori, relative=True)
        for i in range(num_objects):
            for _ in range(5000):
                dx, dy = random_point_in_circle(radius=(0, self._spawn_radius))
                x, y = self._spawn_loc[0] + dx, self._spawn_loc[1] + dy
                #x, y =  dx, dy
                if self.is_valid_spawn_loc(x, y):
                    break
            #self.interface.move_object(self.objects_id[i], [x, y, 0.015],  ori=np.random.uniform(0, 2.0 * np.pi), relative=True)
            self.interface.move_object(self.objects_id[i], [x, y, 0.045], np.random.random(4), relative=True)

        
        for i in range(num_objects, self._max_objects):
            self.interface.move_object(self.objects_id[i], [self.extent + np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0.01])
            
        self.interface.do_steps(100)
    
    @property
    def num_objects(self):
        return self._max_objects

    @property
    def extent(self):
        return 10

class GraspingRoom2(Room):
    def __init__(self, interface, params):
        defaults = dict(
            num_objects=3,
            object_name="greensquareball",
            robot_pos=[0, 0],
        )
        defaults.update(params)
        super().__init__(interface, defaults)

        self.interface.remove_object(self.interface.plane_id)

        self.floors = []
        # starting room
        self.floors.append(FloorPatch(interface, [-1.0, 0.0, 0.0], "floor_carpet_1", 1.0))
        self.floors.append(FloorPatch(interface, [0.0, 0.0, 0.0], "floor_wood_2", 1.0))
        self.floors.append(FloorPatch(interface, [1.0, 0.0, 0.0], "floor_marble_2", 1.0))
        self.floors.append(FloorPatch(interface, [2.0, 0.0, 0.0], "floor_wood_3", 1.0))

        self.floors.append(FloorPatch(interface, [-1.0, 1.0, 0.0], "floor_carpet_5", 1.0))
        self.floors.append(FloorPatch(interface, [0.0, 1.0, 0.0], "floor_wood_1", 1.0))
        self.floors.append(FloorPatch(interface, [1.0, 1.0, 0.0], "floor_carpet_6", 1.0))
        self.floors.append(FloorPatch(interface, [2.0, 1.0, 0.0], "floor_carpet_7", 1.0))


        self.floors.append(FloorPatch(interface, [-1.0,-1.0, 0.0], "floor_marble_1", 1.0))
        self.floors.append(FloorPatch(interface, [0.0, -1.0, 0.0], "floor_carpet_3", 1.0))
        self.floors.append(FloorPatch(interface, [1.0, -1.0, 0.0], "floor_marble_3", 1.0))
        self.floors.append(FloorPatch(interface, [2.0,-1.0, 0.0], "floor_carpet_2", 1.0))

        self.floors.append(FloorPatch(interface, [-1.0, 2.0, 0.0], "floor_carpet_7", 1.0))
        self.floors.append(FloorPatch(interface, [0.0, 2.0, 0.0], "floor_carpet_1", 1.0))
        self.floors.append(FloorPatch(interface, [1.0, 2.0, 0.0], "floor_marble_2", 1.0))
        self.floors.append(FloorPatch(interface, [2.0, 2.0, 0.0], "floor_carpet_4", 1.0))


        # object discard box
        self.discard_walls = []
        self.discard_walls.append(Wall(interface, [10.5, 0, 0], [-1.0, 0.0], 1.0))
        self.discard_walls.append(Wall(interface, [9.5, 0, 0], [1.0, 0.0], 1.0))
        self.discard_walls.append(Wall(interface, [10, 0.5, 0], [0.0, -1.0], 1.0))
        self.discard_walls.append(Wall(interface, [10, -0.5, 0], [0.0, 1.0], 1.0))
        self.discard_floor = FloorPatch(interface, [10.0, 0.0, 0.0], "floor_wood_3", 1.0)

        # objects
        self._num_objects = self.params["num_objects"]
        for i in range(self._num_objects):
            self.objects_id.append(self.interface.spawn_object(URDF[self.params["object_name"]], self.object_discard_pos))

        self.robot_pos = self.params["robot_pos"]

    @property
    def object_discard_pos(self):
        return [10 + np.random.uniform(-0.45, 0.45), np.random.uniform(-0.45, 0.45), np.random.uniform(0.1, 0.5)]

    @property
    def num_objects(self):
        return self._num_objects

    def reset(self):
        for i in range(self._num_objects):
            self.interface.move_object(self.objects_id[i], self.object_discard_pos)

    def random_robot_pos_yaw(self):
        x = np.random.uniform(-1.0, 2.0)
        y = np.random.uniform(-1.0, 2.0)
        yaw = np.random.uniform(0, np.pi * 2.0)
        return np.array([x, y]), yaw


class SimpleRoomWithObstacles(SimpleRoom):
    """ Simple room that has a wall and objects inside, with simple immovable obstacles (not randomly generated). """
    def __init__(self, interface, params):
        defaults = dict()
        super().__init__(interface, defaults)

        # don't spawn in 1m radius around the robot
        self.no_spawn_zones = []
        self.no_spawn_zones.append(lambda x, y: is_in_circle(x, y, 0, 0, 1.0))

        # add 4 rectangular pillars to the 4 corners
        c = self._wall_size / 4
        pillar_size = 0.5

        self.obstacles_id.append(self.interface.spawn_object(URDF["rectangular_pillar"], pos=[c, c, 0], scale=pillar_size))
        self.obstacles_id.append(self.interface.spawn_object(URDF["rectangular_pillar"], pos=[-c, c, 0], scale=pillar_size))
        self.obstacles_id.append(self.interface.spawn_object(URDF["rectangular_pillar"], pos=[c, -c, 0], scale=pillar_size))
        self.obstacles_id.append(self.interface.spawn_object(URDF["rectangular_pillar"], pos=[-c, -c, 0], scale=pillar_size))

        psh = pillar_size * 0.5
        self.no_spawn_zones.append(lambda x, y: is_in_rect(x, y,  c - psh,  c - psh,  c + psh,  c + psh))
        self.no_spawn_zones.append(lambda x, y: is_in_rect(x, y, -c - psh,  c - psh, -c + psh,  c + psh))
        self.no_spawn_zones.append(lambda x, y: is_in_rect(x, y,  c - psh, -c - psh,  c + psh, -c + psh))
        self.no_spawn_zones.append(lambda x, y: is_in_rect(x, y, -c - psh, -c - psh, -c + psh, -c + psh))

        # add 4 short boxes
        box_size = 0.25
        box_height = 0.2
        bsh = box_size * 0.5
        self.obstacles_id.append(self.interface.spawn_object(URDF["solid_box"], pos=[ c,  0, box_height - box_size], scale=box_size))
        self.obstacles_id.append(self.interface.spawn_object(URDF["solid_box"], pos=[-c,  0, box_height - box_size], scale=box_size))
        self.obstacles_id.append(self.interface.spawn_object(URDF["solid_box"], pos=[ 0,  c, box_height - box_size], scale=box_size))
        self.obstacles_id.append(self.interface.spawn_object(URDF["solid_box"], pos=[ 0, -c, box_height - box_size], scale=box_size))

        self.no_spawn_zones.append(lambda x, y: is_in_rect(x, y,  c - bsh,     -bsh,  c + bsh,      bsh))
        self.no_spawn_zones.append(lambda x, y: is_in_rect(x, y, -c - bsh,     -bsh, -c + bsh,      bsh))
        self.no_spawn_zones.append(lambda x, y: is_in_rect(x, y,     -bsh,  c - bsh,      bsh,  c + bsh))
        self.no_spawn_zones.append(lambda x, y: is_in_rect(x, y,     -bsh, -c - bsh,      bsh, -c + bsh))

    def is_valid_spawn_loc(self, x, y):
        for no_spawn in self.no_spawn_zones:
            if no_spawn(x, y):
                return False
        return True

class MediumRoom(Room):
    """ Simple room that has a wall and objects inside, with simple immovable obstacles (not randomly generated). """
    def __init__(self, interface, params):
        defaults = dict(
            num_objects=100, 
            object_name="greensquareball", 
            no_spawn_radius=1.0,
            wall_size=5.0,
        )
        defaults.update(params)
        super().__init__(interface, defaults)

        self._wall_size = self.params["wall_size"]
        self.wall_id = self.interface.spawn_object(URDF["walls_2"], scale=self._wall_size)

        self._num_objects = self.params["num_objects"]
        for i in range(self._num_objects):
            self.objects_id.append(self.interface.spawn_object(URDF[self.params["object_name"]], np.array([0.0, 0.0, 10 + i])))

        # don't spawn in 1m radius around the robot
        self.no_spawn_zones = []
        self._no_spawn_radius = self.params["no_spawn_radius"]
        self.no_spawn_zones.append(lambda x, y: is_in_circle(x, y, 0, 0, self._no_spawn_radius))

        self.interface.change_floor_texture("wood")

        boxes_config = [
            [[-2.259, 2.217, 0], 0, 0.219, "navy"],
            [[-1.776, -1.660, 0], 0, 0.383, "crate"],
            [[-1.12614, -2.08627, 0], 0, 0.321815, "crate"],
            [[-1.31922, 0.195723, 0], 0, 0.270704, "red"],
            [[1.39269, -2.35207, 0], 0, 0.577359, "marble"],
            [[0.33328, -0.75906, 0], 0, 0.233632, "navy"],
            [[2.08005, 1.24189, 0], 0, 0.361638, "crate"],
            [[-1.1331, 2.21413, 0], 0, 0.270704, "red"],
            [[-0.591208, 2.22201, 0], 0, 0.270704, "marble"],
            [[2.13627, 2.1474, 0], 0, 0.270704, "marble"],
            [[1.15959, 0.701976, 0], 0, 0.162524, "navy"],
            [[1.05702, 0.952707, 0], 0, 0.131984, "red"],
        ]

        self.static_objects = []
        for bc in boxes_config:
            self.static_objects.append(TexturedBox(self.interface, *bc[:2], bc[2]*2, texture_name=bc[3]))

    def is_valid_spawn_loc(self, x, y):
        for no_spawn in self.no_spawn_zones:
            if no_spawn(x, y):
                return False
        for so in self.static_objects:
            if so.is_point_inside(x, y):
                return False
        return True

    def reset(self):
        for i in range(self._num_objects):
            for _ in range(5000):
                x, y = np.random.uniform(-self._wall_size * 0.5, self._wall_size * 0.5, size=(2,))
                if self.is_valid_spawn_loc(x, y):
                    break
            self.interface.move_object(self.objects_id[i], [x, y, 0.02])
    
    @property
    def num_objects(self):
        return self._num_objects

    @property
    def extent(self):
        return self._wall_size * 2.0
