import numpy as np

from .utils import *

class StaticObject:
    def __init__(self, interface, pos, rot, scale):
        self.interface = interface
        self.pos = pos
        self.rot = rot
        self.scale = scale

    def is_point_inside(self, x, y):
        return False

class TexturedBox(StaticObject):
    def __init__(self, *args, texture_name=None):
        super().__init__(*args)
        self.id = self.interface.spawn_object(URDF["textured_box"], pos=self.pos, ori=0, scale=self.scale)
        self.interface.p.changeVisualShape(self.id, -1, textureUniqueId=self.interface.load_texture(texture_name))

    def is_point_inside(self, x, y):
        return is_in_rect(x, y,  
            self.pos[0] - self.scale * 0.5,
            self.pos[1] - self.scale * 0.5,
            self.pos[0] + self.scale * 0.5,
            self.pos[1] + self.scale * 0.5,)


class Wall:
    def __init__(self, interface, pos, facing, scale, detection_depth=0.3, is_thin=False):
        self.interface = interface
        self.pos = pos
        self.facing = facing
        self.ori = np.arctan2(facing[1], facing[0])
        self.scale = scale
        self.detection_depth = detection_depth
        self.is_thin = is_thin
        urdf = URDF["wall_single_thin"] if is_thin else URDF["wall_single"]
        self.id = self.interface.spawn_object(urdf, pos=self.pos, ori=self.ori, scale=self.scale)

    def in_turn_detection_box(self, x, y):
        if self.facing == [1, 0]:
            return is_in_rect(x, y, self.pos[0], self.pos[1] - 0.5 * self.scale, self.pos[0] + self.detection_depth, self.pos[1] + 0.5 * self.scale)
        elif self.facing == [-1, 0]:
            return is_in_rect(x, y, self.pos[0] - self.detection_depth, self.pos[1] - 0.5 * self.scale, self.pos[0], self.pos[1] + 0.5 * self.scale)
        elif self.facing == [0, 1]:
            return is_in_rect(x, y, self.pos[0] - 0.5 * self.scale, self.pos[1], self.pos[0] + 0.5 * self.scale, self.pos[1] + self.detection_depth)
        elif self.facing == [0, -1]:
            return is_in_rect(x, y, self.pos[0] - 0.5 * self.scale, self.pos[1] - self.detection_depth, self.pos[0] + 0.5 * self.scale, self.pos[1])
        else:
            print("WARNING", self.facing, "is not a valid facing direction")
            return False


class FloorPatch:
    wall_extent = 0.4
    def __init__(self, interface, pos, texture_name, scale, 
                pos_x_wall=False, pos_y_wall=False, neg_x_wall=False, neg_y_wall=False):
        self.interface = interface
        self.pos = pos
        self.texture_name = texture_name
        self.scale = scale
        self.id = self.interface.spawn_object(URDF["floor_patch"], pos=self.pos, ori=0, scale=self.scale)
        self.interface.p.changeVisualShape(self.id, -1, textureUniqueId=self.interface.load_texture(texture_name))
        self.pos_x_extent = FloorPatch.wall_extent if pos_x_wall else 0.5
        self.pos_y_extent = FloorPatch.wall_extent if pos_y_wall else 0.5
        self.neg_x_extent = FloorPatch.wall_extent if neg_x_wall else 0.5
        self.neg_y_extent = FloorPatch.wall_extent if neg_y_wall else 0.5

    def get_random_loc(self):
        return [np.random.uniform(-self.neg_x_extent, self.pos_x_extent) + self.pos[0], 
                np.random.uniform(-self.neg_y_extent, self.pos_y_extent) + self.pos[1]]

    def is_in_bound(self, x, y):
        return (self.pos[0] - self.neg_x_extent <= x <= self.pos[0] + self.pos_x_extent and \
                self.pos[1] - self.neg_y_extent <= y <= self.pos[1] + self.pos_y_extent)