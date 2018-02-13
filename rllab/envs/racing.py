import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import colorize, seeding

import os
import pyglet
GX = ('DISPLAY' in os.environ.keys())
if GX:
    from gym.envs.classic_control import rendering
    from pyglet.gl import *

# Easiest continuous control task to learn from pixels, a top-down racing environment.
# Discreet control is reasonable in this environment as well, on/off discretisation is
# fine.
#
# State consists of STATE_W x STATE_H pixels.
#
# Reward is -0.1 every frame and +1000/N for every track tile visited, where N is
# the total number of tiles in track. For example, if you have finished in 732 frames,
# your reward is 1000 - 0.1*732 = 926.8 points.
#
# Game is solved when agent consistently gets 900+ points. Track is random every episode.
#
# Episode finishes when all tiles are visited. Car also can go outside of PLAYFIELD, that
# is far off the track, then it will get -100 and die.
#
# Some indicators shown at the bottom of the window and the state RGB buffer. From
# left to right: true speed, four ABS sensors, steering wheel position, gyroscope.
#
# To play yourself (it's rather fast for humans), type:
#
# python gym/envs/box2d/car_racing.py
#
# Remember it's powerful rear-wheel drive car, don't press accelerator and turn at the
# same time.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

STATE_W = 96   # less than Atari 160x192
STATE_H = 96
#STATE_W = 64
#STATE_H = 64
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1200
WINDOW_H = 1000

SCALE       = 6.0        # Track scale
TRACK_RAD   = 900/SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD   = 2000/SCALE # Game over boundary
FPS         = 50
ZOOM        = 2.7        # Camera zoom
ZOOM_FOLLOW = True       # Set to False for fixed view (don't use zoom)

#TRACK_TURN_RATE = 0.31
#TRACK_WIDTH=80/SCALE
#WHEEL_MOMENT_OF_INERTIA = 4000*SIZE*SIZE
#FRICTION_LIMIT          = 1000000*SIZE*SIZE     # friction ~= mass ~= size^2 (calculated implicitly using density)

TRACK_DETAIL_STEP = 21/SCALE
#TRACK_TURN_RATE = 0.31
#TRACK_TURN_RATE = 1.5

#TRACK_WIDTH = 40/SCALE
#TRACK_WIDTH=80/SCALE
BORDER = 8/SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]

SIZE = 0.02
#ENGINE_POWER            = 100000000*SIZE*SIZE
#ENGINE_POWER            = 100000000*SIZE*SIZE

#WHEEL_MOMENT_OF_INERTIA = 4000*SIZE*SIZE
#WHEEL_MOMENT_OF_INERTIA = 8000*SIZE*SIZE

#FRICTION_LIMIT          = 1000000*SIZE*SIZE     # friction ~= mass ~= size^2 (calculated implicitly using density)
#FRICTION_LIMIT          = 1000000*SIZE*SIZE     # friction ~= mass ~= size^2 (calculated implicitly using density)

WHEEL_R  = 27
WHEEL_W  = 14
WHEELPOS = [
    (-55,+80), (+55,+80),
    (-55,-82), (+55,-82)
    ]
HULL_POLY1 =[
    (-60,+130), (+60,+130),
    (+60,+110), (-60,+110)
    ]
HULL_POLY2 =[
    (-15,+120), (+15,+120),
    (+20, +20), (-20,  20)
    ]
HULL_POLY3 =[
    (+25, +20),
    (+50, -10),
    (+50, -40),
    (+20, -90),
    (-20, -90),
    (-50, -40),
    (-50, -10),
    (-25, +20)
    ]
HULL_POLY4 =[
    (-50,-120), (+50,-120),
    (+50,-90),  (-50,-90)
    ]
WHEEL_COLOR = (0.0,0.0,0.0)
WHEEL_WHITE = (0.3,0.3,0.3)
MUD_COLOR   = (0.4,0.4,0.0)

class Car:
    def __init__(self, world, init_angle, init_x, init_y, env_params):
        self.env_params = env_params
        self.world = world
        self.hull = self.world.CreateDynamicBody(
            position = (init_x, init_y),
            angle = init_angle,
            fixtures = [
                fixtureDef(shape = polygonShape(vertices=[ (x*SIZE,y*SIZE) for x,y in HULL_POLY1 ]), density=1.0),
                fixtureDef(shape = polygonShape(vertices=[ (x*SIZE,y*SIZE) for x,y in HULL_POLY2 ]), density=1.0),
                fixtureDef(shape = polygonShape(vertices=[ (x*SIZE,y*SIZE) for x,y in HULL_POLY3 ]), density=1.0),
                fixtureDef(shape = polygonShape(vertices=[ (x*SIZE,y*SIZE) for x,y in HULL_POLY4 ]), density=1.0)
                ]
            )
        self.hull.color = (0.8,0.0,0.0)
        self.wheels = []
        self.wheel_on_grass = []
        self.fuel_spent = 0.0
        WHEEL_POLY = [
            (-WHEEL_W,+WHEEL_R), (+WHEEL_W,+WHEEL_R),
            (+WHEEL_W,-WHEEL_R), (-WHEEL_W,-WHEEL_R)
            ]
        for wx,wy in WHEELPOS:
            front_k = 1.0 if wy > 0 else 1.0
            w = self.world.CreateDynamicBody(
                position = (init_x+wx*SIZE, init_y+wy*SIZE),
                angle = init_angle,
                fixtures = fixtureDef(
                    shape=polygonShape(vertices=[ (x*front_k*SIZE,y*front_k*SIZE) for x,y in WHEEL_POLY ]),
                    density=0.1,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0)
                    )
            w.wheel_rad = front_k*WHEEL_R*SIZE
            w.color = WHEEL_COLOR
            w.gas   = 0.0
            w.brake = 0.0
            w.steer = 0.0
            w.phase = 0.0  # wheel angle
            w.omega = 0.0  # angular velocity
            w.skid_start = None
            w.skid_particle = None
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=w,
                localAnchorA=(wx*SIZE,wy*SIZE),
                localAnchorB=(0,0),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=180*900*SIZE*SIZE,
                motorSpeed = 0,
                lowerAngle = -0.4,
                upperAngle = +0.4,
                )
            w.joint = self.world.CreateJoint(rjd)
            w.tiles = set()
            w.userData = w
            self.wheels.append(w)
        self.drawlist =  self.wheels + [self.hull]
        self.particles = []
        self.wheel_on_grass = [False] * len(self.wheels)

    def gas(self, gas):
        'control: rear wheel drive'
        gas = np.clip(gas, 0, 1)
        for w in self.wheels[2:4]:
            diff = gas - w.gas
            if diff > 0.1: diff = 0.1  # gradually increase, but stop immediately
            w.gas += diff

    def brake(self, b):
        'control: brake b=0..1, more than 0.9 blocks wheels to zero rotation'
        for w in self.wheels:
            w.brake = b

    def steer(self, s):
        'control: steer s=-1..1, it takes time to rotate steering wheel from side to side, s is target position'
        self.wheels[0].steer = s
        self.wheels[1].steer = s

    def step(self, dt):
        for i, w in enumerate(self.wheels):
            # Steer each wheel
            dir = np.sign(w.steer - w.joint.angle)
            val = abs(w.steer - w.joint.angle)
            w.joint.motorSpeed = dir*min(50.0*val, 3.0)

            # Position => friction_limit
            grass = True
            friction_limit = self.env_params['friction_limit']*self.env_params['grass_friction']  # Grass friction if no tile
            for tile in w.tiles:
                friction_limit = max(friction_limit, self.env_params['friction_limit']*self.env_params['road_friction'])
                grass = False

            # Force
            forw = w.GetWorldVector( (0,1) )
            side = w.GetWorldVector( (1,0) )
            v = w.linearVelocity
            vf = forw[0]*v[0] + forw[1]*v[1]  # forward speed
            vs = side[0]*v[0] + side[1]*v[1]  # side speed

            # WHEEL_MOMENT_OF_INERTIA*np.square(w.omega)/2 = E -- energy
            # WHEEL_MOMENT_OF_INERTIA*w.omega * domega/dt = dE/dt = W -- power
            # domega = dt*W/WHEEL_MOMENT_OF_INERTIA/w.omega
            engine_power = self.env_params['engine_power']
            wheel_moment_of_inertia = self.env_params['wheel_moment_of_inertia']
            w.omega += dt*engine_power*w.gas/wheel_moment_of_inertia/(abs(w.omega)+5.0)  # small coef not to divide by zero
            self.fuel_spent += dt*engine_power*w.gas

            if w.brake >= 0.9:
                w.omega = 0
            elif w.brake > 0:
                brake_force = self.env_params['brake_force']    # radians per second
                dir = -np.sign(w.omega)
                val = brake_force*w.brake
                if abs(val) > abs(w.omega): val = abs(w.omega)  # low speed => same as = 0
                w.omega += dir*val
            w.phase += w.omega*dt

            vr = w.omega*w.wheel_rad  # rotating wheel speed
            f_force = -vf + vr        # force direction is direction of speed difference
            p_force = -vs

            # Physically correct is to always apply friction_limit until speed is equal.
            # But dt is finite, that will lead to oscillations if difference is already near zero.
            f_force *= 205000*SIZE*SIZE  # Random coefficient to cut oscillations in few steps (have no effect on friction_limit)
            p_force *= 205000*SIZE*SIZE
            force = np.sqrt(np.square(f_force) + np.square(p_force))

            # Skid trace
            if abs(force) > 2.0*friction_limit:
                if w.skid_particle and w.skid_particle.grass==grass and len(w.skid_particle.poly) < 30:
                    w.skid_particle.poly.append( (w.position[0], w.position[1]) )
                elif w.skid_start is None:
                    w.skid_start = w.position
                else:
                    w.skid_particle = self._create_particle( w.skid_start, w.position, grass )
                    w.skid_start = None
            else:
                w.skid_start = None
                w.skid_particle = None

            if abs(force) > friction_limit:
                f_force /= force
                p_force /= force
                force = friction_limit  # Correct physics here
                f_force *= force
                p_force *= force

            w.omega -= dt*f_force*w.wheel_rad/self.env_params['wheel_moment_of_inertia']

            w.ApplyForceToCenter( (
                p_force*side[0] + f_force*forw[0],
                p_force*side[1] + f_force*forw[1]), True )
            self.wheel_on_grass[i] = grass

    def draw(self, viewer, draw_particles=True):
        if draw_particles:
            for p in self.particles:
                viewer.draw_polyline(p.poly, color=p.color, linewidth=5)
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans*v for v in f.shape.vertices]
                viewer.draw_polygon(path, color=obj.color)
                if "phase" not in obj.__dict__: continue
                a1 = obj.phase
                a2 = obj.phase + 1.2  # radians
                s1 = math.sin(a1)
                s2 = math.sin(a2)
                c1 = math.cos(a1)
                c2 = math.cos(a2)
                if s1>0 and s2>0: continue
                if s1>0: c1 = np.sign(c1)
                if s2>0: c2 = np.sign(c2)
                white_poly = [
                    (-WHEEL_W*SIZE, +WHEEL_R*c1*SIZE), (+WHEEL_W*SIZE, +WHEEL_R*c1*SIZE),
                    (+WHEEL_W*SIZE, +WHEEL_R*c2*SIZE), (-WHEEL_W*SIZE, +WHEEL_R*c2*SIZE)
                    ]
                viewer.draw_polygon([trans*v for v in white_poly], color=WHEEL_WHITE)

    def _create_particle(self, point1, point2, grass):
        class Particle:
            pass
        p = Particle()
        p.color = WHEEL_COLOR if not grass else MUD_COLOR
        p.ttl = 1
        p.poly = [(point1[0],point1[1]), (point2[0],point2[1])]
        p.grass = grass
        self.particles.append(p)
        while len(self.particles) > 30:
            self.particles.pop(0)
        return p

    def destroy(self):
        self.world.DestroyBody(self.hull)
        self.hull = None
        for w in self.wheels:
            self.world.DestroyBody(w)
        self.wheels = []

class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        self._contact(contact, True)
    def EndContact(self, contact):
        self._contact(contact, False)
    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj  = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj  = u1
        if not tile: return

        tile.color[0] = ROAD_COLOR[0]
        tile.color[1] = ROAD_COLOR[1]
        tile.color[2] = ROAD_COLOR[2]
        if not obj or "tiles" not in obj.__dict__: return
        if begin:
            obj.tiles.add(tile)
            #print tile.road_friction, "ADD", len(obj.tiles)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0/len(self.env.track)
                self.env.tile_visited_count += 1
        else:
            obj.tiles.remove(tile)
            #print tile.road_friction, "DEL", len(obj.tiles) -- should delete to zero when on grass (this works)

class CarRacing(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second' : FPS
    }

    def __init__(self, mode='pixels', features= ["pos","vel","abs","hull_ang","wheel_ang",
                                                 "reward","hull_ang_vel","speed","xy_dist_from_road",
                                                 "curves","on_grass","lane_offset","heading_angle"],
                                      track_turn_rate = 0.31,
                                      track_width=40,
                                      friction_limit=1000000,
                                      wheel_moment_of_inertia=4000,
                                      engine_power=100000000,
                                      brake_force=15,
                                      road_friction = 1.0,
                                      grass_friction = 0.6,
                                      #normalize = True,
                                      **kwargs):
        # env params
        self.env_params = {
            "track_turn_rate": track_turn_rate,
            "track_width": track_width / SCALE,
            "friction_limit": friction_limit * SIZE * SIZE,
            "wheel_moment_of_inertia": wheel_moment_of_inertia * SIZE * SIZE,
            "grass_friction": grass_friction,
            "road_friction": road_friction,
            'engine_power': engine_power*SIZE*SIZE,
            'brake_force':brake_force,
        }
        self.normalize = True

        assert mode in ['pixels','state', 'state_action']
        self.mode = mode
        self._seed()
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0,0), contactListener=self.contactListener_keepref)
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.features = features

        n_feat_dict = {"pos":2,"vel":2,"abs":4,"wheel_ang":4,"hull_ang":1,
                       "reward":1,"hull_ang_vel":1,"speed":1,"xy_dist_from_road":2,
                       "curves":3*2, "on_grass": 4, "lane_offset":1,
                       "heading_angle":1}
        n_dim = np.sum([n_feat_dict[feat] for feat in features])

        self.action_space = spaces.Box( np.array([-1,0,0]), np.array([+1,+1,+1]))  # steer, gas, brake
        if self.mode == 'pixels':
            self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3))
        elif self.mode == 'state':
            self.observation_space = spaces.Box(low=np.array([-np.inf for _ in range(n_dim)]),
                                                high=np.array([np.inf for _ in range(n_dim)]))
        elif self.mode == 'state_action':
            self.observation_space = spaces.Box(low=np.array([-np.inf for _ in range(n_dim + 3)]),
                                                high=np.array([np.inf for _ in range(n_dim + 3)]))
        else:
            raise NotImplementedError

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.road: return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        self.car.destroy()

    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            alpha = 2*math.pi*c/CHECKPOINTS + self.np_random.uniform(0, 2*math.pi*1/CHECKPOINTS)
            rad = self.np_random.uniform(TRACK_RAD/3, TRACK_RAD)
            if c==0:
                alpha = 0
                rad = 1.5*TRACK_RAD
            if c==CHECKPOINTS-1:
                alpha = 2*math.pi*c/CHECKPOINTS
                self.start_alpha = 2*math.pi*(-0.5)/CHECKPOINTS
                rad = 1.5*TRACK_RAD
            checkpoints.append( (alpha, rad*math.cos(alpha), rad*math.sin(alpha)) )

        #print "\n".join(str(h) for h in checkpoints)
        #self.road_poly = [ (    # uncomment this to see checkpoints
           #[ (tx,ty) for a,tx,ty in checkpoints ],
           #(0.7,0.7,0.9) ) ]
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5*TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while 1:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2*math.pi
            while True: # Find destination from checkpoints
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0: break
                if not failed: break
                alpha -= 2*math.pi
                continue
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            proj = r1x*dest_dx + r1y*dest_dy  # destination vector projected on rad
            while beta - alpha >  1.5*math.pi: beta -= 2*math.pi
            while beta - alpha < -1.5*math.pi: beta += 2*math.pi
            prev_beta = beta
            proj *= SCALE
            if proj >  0.3: beta -= min(self.env_params['track_turn_rate'], abs(0.001*proj))
            if proj < -0.3: beta += min(self.env_params['track_turn_rate'], abs(0.001*proj))
            x += p1x*TRACK_DETAIL_STEP
            y += p1y*TRACK_DETAIL_STEP
            track.append( (alpha,prev_beta*0.5 + beta*0.5,x,y) )
            if laps > 4: break
            no_freeze -= 1
            if no_freeze==0: break
        #print "\n".join([str(t) for t in enumerate(track)])

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i==0: return False  # Failed
            pass_through_start = track[i][0] > self.start_alpha and track[i-1][0] <= self.start_alpha
            if pass_through_start and i2==-1:
                i2 = i
            elif pass_through_start and i1==-1:
                i1 = i
                break
        #print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2-i1))
        assert i1!=-1
        assert i2!=-1

        track = track[i1:i2-1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square( first_perp_x*(track[0][2] - track[-1][2]) ) +
            np.square( first_perp_y*(track[0][3] - track[-1][3]) ))
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False]*len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i-neg-0][1]
                beta2 = track[i-neg-1][1]
                good &= abs(beta1 - beta2) > self.env_params['track_turn_rate']*0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i-neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i-1]
            track_width = self.env_params['track_width']
            road1_l = (x1 - track_width*math.cos(beta1), y1 - track_width*math.sin(beta1))
            road1_r = (x1 + track_width*math.cos(beta1), y1 + track_width*math.sin(beta1))
            road2_l = (x2 - track_width*math.cos(beta2), y2 - track_width*math.sin(beta2))
            road2_r = (x2 + track_width*math.cos(beta2), y2 + track_width*math.sin(beta2))
            t = self.world.CreateStaticBody( fixtures = fixtureDef(
                shape=polygonShape(vertices=[road1_l, road1_r, road2_r, road2_l])
                ))
            t.userData = t
            c = 0.01*(i%3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = False
            t.road_friction = self.env_params['road_friction']
            t.fixtures[0].sensor = True
            self.road_poly.append(( [road1_l, road1_r, road2_r, road2_l], t.color ))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (x1 + side* track_width        *math.cos(beta1), y1 + side* track_width        *math.sin(beta1))
                b1_r = (x1 + side*(track_width+BORDER)*math.cos(beta1), y1 + side*(track_width+BORDER)*math.sin(beta1))
                b2_l = (x2 + side* track_width        *math.cos(beta2), y2 + side* track_width        *math.sin(beta2))
                b2_r = (x2 + side*(track_width+BORDER)*math.cos(beta2), y2 + side*(track_width+BORDER)*math.sin(beta2))
                self.road_poly.append(( [b1_l, b1_r, b2_r, b2_l], (1,1,1) if i%2==0 else (1,0,0) ))
        self.track = track

        return True

    def _reset(self):
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []
        self.human_render = False

        while True:
            success = self._create_track()
            if success: break
            print("retry to generate track (normal if there are not many of this messages)")
        self.car = Car(self.world, *self.track[0][1:4], env_params = self.env_params)

        return self._step(None)[0]

    def _featurize(self):
        info = {}

        # Added this
        on_grass = self.car.step(1.0/FPS)
        track_xy = [(x,y) for a,b,x,y in self.track] # do I know current track index?
        track_ab = np.row_stack([(a,b) for a,b,x,y in self.track])
        curvature = np.roll(track_ab[:,1],1) - track_ab[:,1]

        XY = np.row_stack(track_xy)
        D = XY - np.array(self.car.hull.position)
        dists = np.sqrt(np.sum(np.square(D),axis=1))
        curr_tile = np.argmin(dists)
        xy_dist_from_road = D[curr_tile]

        T = len(self.track)
        lane_center =  XY[(curr_tile + 1) % T] - XY[(curr_tile - 1) % T] # direction vector of lane ...
        car_displace = np.array(self.car.hull.position) - XY[(curr_tile - 1) % T] # displacement of car position from position of PREVIOUS tile ...
        #car_displace = car_xy_after - car_xy_before
        car_proj = (np.dot(car_displace, lane_center)/np.dot(lane_center, lane_center)) * lane_center

        lane_offset = np.linalg.norm(car_displace - car_proj)
        similar_to_desired_angle = 1.0 - (np.dot(car_displace, lane_center) /
                (np.linalg.norm(car_displace) * np.linalg.norm(lane_center)))

        #near_curves = np.concatenate([track_ab[curr_tile],
        #                            track_ab[(curr_tile + 1) % len(self.track)],
        #                            track_ab[(curr_tile - 1) % len(self.track)]])
        ssize = 3
        near_curves = np.array([curvature[i % len(self.track)] for i in
            range(curr_tile-(3*ssize),curr_tile+(3*ssize),ssize)])

        def compute_hull_angle(angle,vel):
            if np.linalg.norm(vel) > 0.5:
                angle = math.atan2(vel[0], vel[1])
            return angle
        feature_dict = {"pos":np.array(self.car.hull.position),
                        "vel":np.array(self.car.hull.linearVelocity),
                        "abs":np.array([wheel.omega for wheel in self.car.wheels]),
                        "wheel_ang":np.array([wheel.joint.angle for wheel in self.car.wheels]),
                        "hull_ang_vel":np.array([self.car.hull.angularVelocity]),
                        "reward":np.array([self.reward]),
                        "hull_ang":np.array([compute_hull_angle(self.car.hull.angle,self.car.hull.linearVelocity)]),
                        "speed":np.array([np.sqrt(np.square(self.car.hull.linearVelocity[0]) + np.square(self.car.hull.linearVelocity[1]))]),
                        "xy_dist_from_road":xy_dist_from_road,
                        "on_grass":np.array(self.car.wheel_on_grass).astype('float32'),
                        "curves":near_curves,
                        "lane_offset":np.array([lane_offset]),
                        "heading_angle":np.array([similar_to_desired_angle])}

        if self.normalize:
            feature_dict["vel"] /= 50.
            feature_dict["hull_ang_vel"] /= 50.
            feature_dict["speed"] /= 50.
            xydfr = feature_dict["xy_dist_from_road"]
            feature_dict["xy_dist_from_road"] = (np.sign(xydfr) *
                    np.log(np.abs(xydfr) + 1e-8)) / 3.0

        info['state'] = np.concatenate([feature_dict[feat] for feat in self.features])
        info['labels'] = self.features
        return info

    def _step(self, action):
        info = {}

        car_xy_before = np.array(self.car.hull.position)
        if action is not None:
            action = np.clip(action, self.action_space.low,
                    self.action_space.high)
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])

        self.world.Step(1.0/FPS, 6*30, 2*30)
        car_xy_after = np.array(self.car.hull.position)

        self.t += 1.0/FPS

        info = self._featurize()
        if self.mode == 'pixels' or GX:
            self.image = self._render(mode="state_pixels")
        if self.mode == 'state_action':
            if action is None:
                act = np.zeros(3)
            else:
                act = action
            info['state'] = np.concatenate([info['state'], act])

        step_reward = 0
        done = False
        if action is not None: # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            #self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            #if self.tile_visited_count==len(self.track):
            #    done = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                #done = True
                step_reward = -1.

        if self.mode == 'pixels':
            self.state = self.image
        elif self.mode in ['state','state_action']:
            self.state = info['state']

        return self.state, step_reward, done, info

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label('0000', font_size=36,
                x=20, y=WINDOW_H*2.5/40.00, anchor_x='left', anchor_y='center',
                color=(255,255,255,255))
            self.transform = rendering.Transform()

        if "t" not in self.__dict__: return  # reset() not called yet

        zoom = 0.1*SCALE*max(1-self.t, 0) + ZOOM*SCALE*min(self.t, 1)   # Animate zoom first second
        zoom_state  = ZOOM*SCALE*STATE_W/WINDOW_W
        zoom_video  = ZOOM*SCALE*VIDEO_W/WINDOW_W
        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W/2 - (scroll_x*zoom*math.cos(angle) - scroll_y*zoom*math.sin(angle)),
            WINDOW_H/4 - (scroll_x*zoom*math.sin(angle) + scroll_y*zoom*math.cos(angle)) )
        self.transform.set_rotation(angle)

        self.car.draw(self.viewer, mode!="state_pixels")

        arr = None
        win = self.viewer.window
        if mode != 'state_pixels':
            win.switch_to()
            win.dispatch_events()
            pass

        if mode=="rgb_array" or mode=="state_pixels":
            #win.clear()
            t = self.transform
            if mode=='rgb_array':
                VP_W = VIDEO_W
                VP_H = VIDEO_H
            else:
                VP_W = STATE_W
                VP_H = STATE_H
            glViewport(0, 0, VP_W, VP_H)
            t.enable()
            self._render_road()
            for geom in self.viewer.onetime_geoms:
                geom.render()
            t.disable()
            self._render_indicators(WINDOW_W, WINDOW_H)  # TODO: find why 2x needed, wtf
            image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            arr = arr.reshape(VP_H, VP_W, 4)
            arr = arr[::-1, :, 0:3]

        #if mode=="rgb_array" and not self.human_render: # agent can call or not call env.render() itself when recording video.
            #win.flip()
            #pass

        if mode=="rgb_array" and not self.human_render: # agent can call or not call env.render() itself when recording video.
            win.flip()
            pass

        if mode=='human':
            self.human_render = True
            win.clear()
            t = self.transform
            glViewport(0, 0, WINDOW_W, WINDOW_H)
            t.enable()
            self._render_road()
            for geom in self.viewer.onetime_geoms:
                geom.render()
            t.disable()
            self._render_indicators(WINDOW_W, WINDOW_H)
            win.flip()

        self.viewer.onetime_geoms = []
        return arr

    def _render_road(self):
        glBegin(GL_QUADS)
        glColor4f(0.4, 0.8, 0.4, 1.0)
        glVertex3f(-PLAYFIELD, +PLAYFIELD, 0)
        glVertex3f(+PLAYFIELD, +PLAYFIELD, 0)
        glVertex3f(+PLAYFIELD, -PLAYFIELD, 0)
        glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)
        glColor4f(0.4, 0.9, 0.4, 1.0)
        k = PLAYFIELD/20.0
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                glVertex3f(k*x + k, k*y + 0, 0)
                glVertex3f(k*x + 0, k*y + 0, 0)
                glVertex3f(k*x + 0, k*y + k, 0)
                glVertex3f(k*x + k, k*y + k, 0)
        for poly, color in self.road_poly:
            glColor4f(color[0], color[1], color[2], 1)
            for p in poly:
                glVertex3f(p[0], p[1], 0)
        glEnd()

    def _render_indicators(self, W, H):
        glBegin(GL_QUADS)
        s = W/40.0
        h = H/40.0
        glColor4f(0,0,0,1)
        glVertex3f(W, 0, 0)
        glVertex3f(W, 5*h, 0)
        glVertex3f(0, 5*h, 0)
        glVertex3f(0, 0, 0)
        def vertical_ind(place, val, color):
            glColor4f(color[0], color[1], color[2], 1)
            glVertex3f((place+0)*s, h + h*val, 0)
            glVertex3f((place+1)*s, h + h*val, 0)
            glVertex3f((place+1)*s, h, 0)
            glVertex3f((place+0)*s, h, 0)
        def horiz_ind(place, val, color):
            glColor4f(color[0], color[1], color[2], 1)
            glVertex3f((place+0)*s, 4*h , 0)
            glVertex3f((place+val)*s, 4*h, 0)
            glVertex3f((place+val)*s, 2*h, 0)
            glVertex3f((place+0)*s, 2*h, 0)
        true_speed = np.sqrt(np.square(self.car.hull.linearVelocity[0]) + np.square(self.car.hull.linearVelocity[1]))
        vertical_ind(5, 0.02*true_speed, (1,1,1))
        vertical_ind(7, 0.01*self.car.wheels[0].omega, (0.0,0,1)) # ABS sensors
        vertical_ind(8, 0.01*self.car.wheels[1].omega, (0.0,0,1))
        vertical_ind(9, 0.01*self.car.wheels[2].omega, (0.2,0,1))
        vertical_ind(10,0.01*self.car.wheels[3].omega, (0.2,0,1))
        horiz_ind(20, -10.0*self.car.wheels[0].joint.angle, (0,1,0))
        horiz_ind(30, -0.8*self.car.hull.angularVelocity, (1,0,0))
        glEnd()
        self.score_label.text = "%04i" % self.reward
        self.score_label.draw()


if __name__=="__main__":
    from pyglet.window import key
    a = np.array( [0.0, 0.0, 0.0] )
    def key_press(k, mod):
        global restart
        if k==0xff0d: restart = True
        if k==key.LEFT:  a[0] = -1.0
        if k==key.RIGHT: a[0] = +1.0
        if k==key.UP:    a[1] = +1.0
        if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
    def key_release(k, mod):
        if k==key.LEFT  and a[0]==-1.0: a[0] = 0
        if k==key.RIGHT and a[0]==+1.0: a[0] = 0
        if k==key.UP:    a[1] = 0
        if k==key.DOWN:  a[2] = 0
    env = CarRacing()
    env.render()
    record_video = False
    if record_video:
        env.monitor.start('/tmp/video-test', force=True)
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    while True:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                #import matplotlib.pyplot as plt
                #plt.imshow(s)
                #plt.savefig("test.jpeg")
            steps += 1
            if not record_video: # Faster, but you can as well call env.render() every time to play full window.
                env.render()
            if done or restart: break
    env.monitor.close()
