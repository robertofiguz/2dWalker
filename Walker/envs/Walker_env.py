import gym
from gym import error, spaces, utils, Env
from gym.spaces import Discrete, MultiDiscrete, Box
from gym.utils import seeding
import math
import pymunk
import pygame

screen_width = 1904
screen_height = 960

class Robot():
    def __init__(self, space):
        
        self.tick = 0

        moment = 10
        friction = 0.6

        self.shape = pymunk.Poly.create_box(None, (50, 100))
        body_moment = pymunk.moment_for_poly(moment, self.shape.get_vertices())
        self.body = pymunk.Body(moment, body_moment)
        
        self.body.position = (200, 350)
        self.shape.body = self.body
        self.shape.color = (150, 150, 150, 0)

        head_moment = pymunk.moment_for_circle(moment, 0, 30)
        self.head_body = pymunk.Body(moment, head_moment)
        self.head_body.position = (self.body.position.x, self.body.position.y+80)
        self.head_shape = pymunk.Circle(self.head_body, 30)
        self.head_shape.friction = friction
        self.head_joint = pymunk.PivotJoint(self.head_body, self.body, (-5, -30), (-5, 50))
        self.head_joint2 = pymunk.PivotJoint(self.head_body, self.body, (5, -30), (5, 50))


        arm_size = (100, 20)
        self.left_arm_upper_shape = pymunk.Poly.create_box(None, arm_size)
        left_arm_upper_moment = pymunk.moment_for_poly(moment, self.left_arm_upper_shape.get_vertices())
        self.left_arm_upper_body = pymunk.Body(moment, left_arm_upper_moment)
        self.left_arm_upper_body.position = (self.body.position.x-70, self.body.position.y+30)
        self.left_arm_upper_shape.body = self.left_arm_upper_body
        self.left_arm_upper_joint = pymunk.PivotJoint(self.left_arm_upper_body, self.body, (arm_size[0] / 2, 0), (-25, 30))
        self.la_motor = pymunk.SimpleMotor(self.body, self.left_arm_upper_body, 0)

        self.right_arm_upper_shape = pymunk.Poly.create_box(None, arm_size)
        right_arm_upper_moment = pymunk.moment_for_poly(moment, self.right_arm_upper_shape.get_vertices())
        self.right_arm_upper_body = pymunk.Body(moment, right_arm_upper_moment)
        self.right_arm_upper_body.position = (self.body.position.x+70, self.body.position.y+30)
        self.right_arm_upper_shape.body = self.right_arm_upper_body
        self.right_arm_upper_joint = pymunk.PivotJoint(self.right_arm_upper_body, self.body, (-arm_size[0] / 2, 0), (25, 30))
        self.ra_motor = pymunk.SimpleMotor(self.body, self.right_arm_upper_body, 0)

        thigh_size = (30, 60)
        self.lu_shape = pymunk.Poly.create_box(None, thigh_size)
        lu_moment = pymunk.moment_for_poly(moment, self.lu_shape.get_vertices())
        self.lu_body = pymunk.Body(moment, lu_moment)
        self.lu_body.position = (self.body.position.x-20, self.body.position.y-75)
        self.lu_shape.body = self.lu_body
        self.lu_shape.friction = friction
        self.lu_joint = pymunk.PivotJoint(self.lu_body, self.body, (0, thigh_size[1] / 2), (-20, -50))
        self.lu_motor = pymunk.SimpleMotor(self.body, self.lu_body, 0)

        self.ru_shape = pymunk.Poly.create_box(None, thigh_size)
        ru_moment = pymunk.moment_for_poly(moment, self.ru_shape.get_vertices())
        self.ru_body = pymunk.Body(moment, ru_moment)
        self.ru_body.position = (self.body.position.x+20, self.body.position.y - 75)
        self.ru_shape.body = self.ru_body
        self.ru_shape.friction = friction
        self.ru_joint = pymunk.PivotJoint(self.ru_body, self.body, (0, thigh_size[1] / 2), (20, -50))
        self.ru_motor = pymunk.SimpleMotor(self.body, self.ru_body, 0)

        leg_size = (20, 70)
        self.ld_shape = pymunk.Poly.create_box(None, leg_size)
        ld_moment = pymunk.moment_for_poly(moment, self.ld_shape.get_vertices())
        self.ld_body = pymunk.Body(moment, ld_moment)
        self.ld_body.position = (self.lu_body.position.x, self.lu_body.position.y - 65)
        self.ld_shape.body = self.ld_body
        self.ld_shape.friction = friction
        self.ld_joint = pymunk.PivotJoint(self.ld_body, self.lu_body, (0, leg_size[1] / 2), (0, -thigh_size[1] / 2))
        self.ld_motor = pymunk.SimpleMotor(self.lu_body, self.ld_body, 0)

        self.rd_shape = pymunk.Poly.create_box(None, leg_size)
        rd_moment = pymunk.moment_for_poly(moment, self.rd_shape.get_vertices())
        self.rd_body = pymunk.Body(moment, rd_moment)
        self.rd_body.position = (self.ru_body.position.x, self.ru_body.position.y - 65)
        self.rd_shape.body = self.rd_body
        self.rd_shape.friction = friction
        self.rd_joint = pymunk.PivotJoint(self.rd_body, self.ru_body, (0, leg_size[1] / 2), (0, -thigh_size[1] / 2))
        self.rd_motor = pymunk.SimpleMotor(self.ru_body, self.rd_body, 0)


        foot_size = (45, 20)
        self.lf_shape = pymunk.Poly.create_box(None, foot_size)
        rd_moment = pymunk.moment_for_poly(moment, self.lf_shape.get_vertices())
        self.lf_body = pymunk.Body(moment, rd_moment)
        self.lf_body.position = (self.ld_body.position.x + foot_size[0]/6, self.ld_body.position.y - (foot_size[1]*2))
        self.lf_shape.body = self.lf_body
        self.lf_shape.friction = friction
        self.lf_shape.elasticity = 0.1
        self.lf_joint = pymunk.PivotJoint(self.ld_body, self.lf_body, (-5, -leg_size[1] / 2), (-foot_size[0]/2 + 10, foot_size[1]/2))
        self.lf_motor = pymunk.SimpleMotor(self.ld_body, self.lf_body, 0)

        self.rf_shape = pymunk.Poly.create_box(None, foot_size)
        rd_moment = pymunk.moment_for_poly(moment, self.rf_shape.get_vertices())
        self.rf_body = pymunk.Body(moment, rd_moment)
        self.rf_body.position = (self.rd_body.position.x + foot_size[0]/6, self.rd_body.position.y - (foot_size[1]*2))
        self.rf_shape.body = self.rf_body
        self.rf_shape.friction = friction
        self.rf_shape.elasticity = 0.1
        self.rf_joint = pymunk.PivotJoint(self.rd_body, self.rf_body, (-5, -leg_size[1] / 2), (-foot_size[0]/2 + 10, foot_size[1]/2))
        self.rf_motor = pymunk.SimpleMotor(self.rd_body, self.rf_body, 0)

        space.add(self.body, self.shape, self.head_body, self.head_shape, self.head_joint, self.head_joint2)
        space.add(self.left_arm_upper_body, self.left_arm_upper_shape, self.left_arm_upper_joint, self.la_motor)
        space.add(self.right_arm_upper_body, self.right_arm_upper_shape, self.right_arm_upper_joint, self.ra_motor)
        space.add(self.lu_body, self.lu_shape, self.lu_joint, self.lu_motor)
        space.add(self.ru_body, self.ru_shape, self.ru_joint, self.ru_motor)
        space.add(self.ld_body, self.ld_shape, self.ld_joint, self.ld_motor)
        space.add(self.rd_body, self.rd_shape, self.rd_joint, self.rd_motor)
        space.add(self.lf_body, self.lf_shape, self.lf_joint, self.lf_motor)
        space.add(self.rf_body, self.rf_shape, self.rf_joint, self.rf_motor)


        shape_filter = pymunk.ShapeFilter(group=1)
        self.shape.filter = shape_filter
        self.head_shape.filter = shape_filter
        self.left_arm_upper_shape.filter = shape_filter
        self.right_arm_upper_shape.filter = shape_filter
        self.lu_shape.filter = shape_filter
        self.ru_shape.filter = shape_filter
        self.ld_shape.filter = shape_filter
        self.rd_shape.filter = shape_filter
        self.lf_shape.filter = shape_filter
        self.rf_shape.filter = shape_filter

        self.lu_flag = False
        self.ld_flag = False
        self.ru_flag = False
        self.rd_flag = False
        self.la_flag = False
        self.ra_flag = False
        self.lf_flag = False
        self.rf_flag = False

    def get_data(self):
        lu = ((360 - math.degrees(self.lu_body.angle)) - (360 - math.degrees(self.body.angle))) / 360.0
        ld = ((360 - math.degrees(self.ld_body.angle)) - (360 - math.degrees(self.body.angle))) / 360.0
        lf = ((360 - math.degrees(self.lf_body.angle)) - (360 - math.degrees(self.body.angle))) / 360.0
        ru = ((360 - math.degrees(self.ru_body.angle)) - (360 - math.degrees(self.body.angle))) / 360.0
        rd = ((360 - math.degrees(self.rd_body.angle)) - (360 - math.degrees(self.body.angle))) / 360.0
        rf = ((360 - math.degrees(self.rf_body.angle)) - (360 - math.degrees(self.body.angle))) / 360.0
        la = ((360 - math.degrees(self.left_arm_upper_body.angle)) - (360 - math.degrees(self.body.angle))) / 360.0
        ra = ((360 - math.degrees(self.right_arm_upper_body.angle)) - (360 - math.degrees(self.body.angle))) / 360.0
        return ru, rd, lu, ld, la, ra, lf, rf
        #removed self.body,angle

    def update(self):
        #lu
        self.lu_flag = False
        if (360 - math.degrees(self.lu_body.angle)) - (360 - math.degrees(self.body.angle)) >= 90 and self.lu_motor.rate > 0:
            self.lu_motor.rate = 0
            self.lu_flag = True
        elif (360 - math.degrees(self.lu_body.angle)) - (360 - math.degrees(self.body.angle)) <= -90 and self.lu_motor.rate < 0:
            self.lu_motor.rate = 0
            self.lu_flag = True

        #ld
        self.ld_flag = False
        if (360 - math.degrees(self.ld_body.angle)) - (360 - math.degrees(self.lu_body.angle)) >= 90 and self.ld_motor.rate > 0:
            self.ld_motor.rate = 0
            self.ld_flag = True
        elif (360 - math.degrees(self.ld_body.angle)) - (360 - math.degrees(self.lu_body.angle)) <= -90 and self.ld_motor.rate < 0:
            self.ld_motor.rate = 0
            self.ld_flag = True

        #ru
        self.ru_flag = False
        if (360 - math.degrees(self.ru_body.angle)) - (360 - math.degrees(self.body.angle)) >= 90 and self.ru_motor.rate > 0:
            self.ru_motor.rate = 0
            self.ru_flag = True
        elif (360 - math.degrees(self.ru_body.angle)) - (360 - math.degrees(self.body.angle)) <= -90 and self.ru_motor.rate < 0:
            self.ru_motor.rate = 0
            self.ru_flag = True

        #rd
        self.rd_flag = False
        if (360 - math.degrees(self.rd_body.angle)) - (360 - math.degrees(self.ru_body.angle)) >= 90 and self.rd_motor.rate > 0:
            self.rd_motor.rate = 0
            self.rd_flag = True
        elif (360 - math.degrees(self.rd_body.angle)) - (360 - math.degrees(self.ru_body.angle)) <= -90 and self.rd_motor.rate < 0:
            self.rd_motor.rate = 0
            self.rd_flag = True


        #lf
        self.lf_flag = False
        if (360 - math.degrees(self.lf_body.angle)) - (360 - math.degrees(self.ld_body.angle)) >= 90 and self.lf_motor.rate > 0:
            self.lf_motor.rate = 0
            self.lf_flag = True
        elif (360 - math.degrees(self.lf_body.angle)) - (360 - math.degrees(self.ld_body.angle)) <= -45 and self.lf_motor.rate < 0:
            self.lf_motor.rate = 0
            self.lf_flag = True


        #rf
        self.rf_flag = False
        if (360 - math.degrees(self.rf_body.angle)) - (360 - math.degrees(self.rd_body.angle)) >= 90 and self.rf_motor.rate > 0:
            self.rf_motor.rate = 0
            self.rf_flag = True
        elif (360 - math.degrees(self.rf_body.angle)) - (360 - math.degrees(self.rd_body.angle)) <= -45 and self.rf_motor.rate < 0:
            self.rf_motor.rate = 0
            self.rf_flag = True
    
    def add_land(self,space):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = (0, 100)
        land = pymunk.Segment(body, (0, 50), (99999, 50), 10)
        land.friction = 1.0
        land.elasticity = 0.1
        space.add(body, land)

        body_2 = pymunk.Body(body_type=pymunk.Body.STATIC)
        body_2.position = (100, -50)
        t_block = pymunk.Segment(body_2, (0, 100), (20, 100), 10)
        space.add(body_2, t_block)


class Walker(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = MultiDiscrete([3]*8)
        self.observation_space = Box(-20,20,[8])
        self.viewer = None
        self.last_horizontal_pos = 0
        self.last_vertical_pos = 0
    def check_fall(self):
      if self.robot.body.position[1] < self.initial_height-50:
            return True
      if self.robot.body.position[0] < 0 or self.robot.body.position[0] > screen_width:
            return True
      return False
    def calculate_reward(self):
      shape = self.space.shapes[-2]
      contact_lf = len(self.robot.lf_shape.shapes_collide(b=shape).points)
      contact_rf = len(self.robot.rf_shape.shapes_collide(b=shape).points)

      if (self.robot.body.position[0] - self.last_horizontal_pos) > 1:
        reward = 0
      elif 1 > (self.robot.body.position[0] - self.last_horizontal_pos) > -1:
        reward = -100
      elif (self.robot.body.position[0] - self.last_horizontal_pos) < -1:
        reward = -200
      if not contact_lf and not contact_rf:
        reward -= 50
      return -1 #set to only get reward if reaches target

    def check_complete(self):
      if self.robot.body.position[0] == 100: # 500 is the position of the target
        return True
    def step(self, actions):
        actions = [(a-1)*2 for a in actions]
        self.robot.ru_motor.rate = actions[0]
        self.robot.rd_motor.rate = actions[1]
        self.robot.lu_motor.rate = actions[2]
        self.robot.ld_motor.rate = actions[3]
        self.robot.la_motor.rate = actions[4]
        self.robot.ra_motor.rate = actions[5]
        self.robot.lf_motor.rate = actions[6]
        self.robot.rf_motor.rate = actions[7]

        self.robot.update()
        self.space.step(1/50)


        self.robot.ru_motor.rate = 0
        self.robot.rd_motor.rate = 0
        self.robot.lu_motor.rate = 0
        self.robot.ld_motor.rate = 0
        self.robot.la_motor.rate = 0
        self.robot.ra_motor.rate = 0
        self.robot.lf_motor.rate = 0
        self.robot.rf_motor.rate = 0

        done = False
        reward = self.calculate_reward()

        if self.check_fall():
          done = True
          reward = -1000 # -200 reresents the highest penalty

        if self.check_complete(): 
          done = True  
          reward = 100

        info = {}
        observation = self.robot.get_data()

        self.last_horizontal_pos = self.robot.body.position[0]
        self.last_vertical_pos = self.robot.body.position[1]

        return(
        observation,
        reward,
        done,
        info)

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = pygame.init()
            pymunk.pygame_util.positive_y_is_up = True
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.screen.fill((255, 255, 255))
        self.space.debug_draw(self.draw_options)
        pygame.display.flip()
        self.clock.tick(25)
        return pygame.surfarray.array3d(self.screen)

        
    def reset(self):
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -990)
        self.robot = Robot(self.space)
        self.robot.add_land(self.space)
        self.initial_height = self.robot.body.position[1]
        self.initial_horizontal = self.robot.body.position[0]
        observation = self.robot.get_data()
        return(observation)

