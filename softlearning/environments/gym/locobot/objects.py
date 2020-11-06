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

class ColoredBox(StaticObject):
    def __init__(self, interface, pos, scale, color="grey"):
        super().__init__(interface, pos, 0, scale)
        if color == "grey":
            urdf = URDF["box_grey"]
        elif color == "dark_grey":
            urdf = URDF["box_dark_grey"]
        else:
            raise NotImplementedError(color + " box not implemented")
        self.id = self.interface.spawn_object(urdf, pos=self.pos, ori=0, scale=self.scale)

    def is_point_inside(self, x, y, wiggle=0.03):
        return is_in_rect(x, y,  
            self.pos[0] - self.scale - wiggle,
            self.pos[1] - self.scale - wiggle,
            self.pos[0] + self.scale + wiggle,
            self.pos[1] + self.scale + wiggle)

    def closest_outside_pos(self, x, y):
        rel_x = x - self.pos[0]
        rel_y = y - self.pos[1]

        dist_to_center = max(abs(rel_x), abs(rel_y))

        norm_rel_x = rel_x / dist_to_center
        norm_rel_y = rel_y / dist_to_center

        rel_out_x = norm_rel_x * (self.scale + 0.08)
        rel_out_y = norm_rel_y * (self.scale + 0.08)

        return rel_out_x + self.pos[0], rel_out_y + self.pos[1]

    def is_in_detection(self, x, y, depth=0.3):
        closest_x = np.clip(x, self.pos[0] - self.scale, self.pos[0] + self.scale)
        closest_y = np.clip(y, self.pos[1] - self.scale, self.pos[1] + self.scale)

        dist_sq = (x - closest_x) ** 2 + (y - closest_y) ** 2

        return dist_sq <= depth ** 2

    def get_normal(self, x, y):
        closest_x = np.clip(x, self.pos[0] - self.scale, self.pos[0] + self.scale)
        closest_y = np.clip(y, self.pos[1] - self.scale, self.pos[1] + self.scale)

        diff = np.array([x - closest_x, y - closest_y])
        return diff / np.linalg.norm(diff)


class ColoredCylinder(StaticObject):
    def __init__(self, interface, pos, scale, color="grey"):
        super().__init__(interface, pos, 0, scale)
        if color == "grey":
            urdf = URDF["cylinder_grey"]
        elif color == "black":
            urdf = URDF["cylinder_black"]
        else:
            raise NotImplementedError(color + " cylinder not implemented")
        self.id = self.interface.spawn_object(urdf, pos=self.pos, ori=0, scale=self.scale)

    def is_point_inside(self, x, y, wiggle=0.03):
        return is_in_circle(x, y, self.pos[0], self.pos[1], self.scale + wiggle)

    def closest_outside_pos(self, x, y):
        rel_x = x - self.pos[0]
        rel_y = y - self.pos[1]

        dist_to_center = np.sqrt(rel_x ** 2 + rel_y ** 2)

        norm_rel_x = rel_x / dist_to_center
        norm_rel_y = rel_y / dist_to_center

        rel_out_x = norm_rel_x * (self.scale + 0.08)
        rel_out_y = norm_rel_y * (self.scale + 0.08)

        return rel_out_x + self.pos[0], rel_out_y + self.pos[1]

    def is_in_detection(self, x, y, depth=0.3):
        diff = np.array([x - self.pos[0], y - self.pos[1]])
        dist = np.linalg.norm(diff)
        return dist <= self.scale + depth

    def get_normal(self, x, y):
        diff = np.array([x - self.pos[0], y - self.pos[1]])
        return diff / np.linalg.norm(diff)

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