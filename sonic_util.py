"""
Versão usando homografia
"""

import sys
sys.path.append('/home/felipe/anaconda3/envs/newRL/lib/python3.8/site-packages')

import gym
import numpy as np
import cv2
from PIL import Image
from collections import deque


import retrowrapper #python -m retro.import.sega_classics
from retro_contest.local import make
retrowrapper.set_retro_make( make )
from stable_baselines3.common.monitor import Monitor


  

def make_env(idx, trajectory=None, game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', myReward=True, stack=True, scale_rew=True, allowbacktrace=True, logdir=None):
    
   
    
    """
    Create an environment with some standard wrappers.
    """
    env_idx = idx 
    # size 47
    dicts = [
            {'game': 'SonicTheHedgehog-Genesis', 'state': 'GreenHillZone.Act1'},
            {'game': 'SonicTheHedgehog-Genesis', 'state': 'GreenHillZone.Act2'},
            {'game': 'SonicTheHedgehog-Genesis', 'state': 'ScrapBrainZone.Act1'}, 
            {'game': 'SonicTheHedgehog-Genesis', 'state': 'StarLightZone.Act2'}, 
            {'game': 'SonicTheHedgehog-Genesis', 'state': 'StarLightZone.Act3'},
            {'game': 'SonicTheHedgehog-Genesis', 'state': 'SpringYardZone.Act1'},            
    ]
    
   
    env = make(game=dicts[env_idx]['game'], state=dicts[env_idx]['state'], bk2dir="./records")
    env = SonicDiscretizer(env)
    env = PreprocessFrame(env)
    env = Monitor(env, logdir)

    
    if myReward:
        env = CalcReward(env)
    if scale_rew:
        env = RewardScaler(env)
#    if allowbacktrace:
#        env = AllowBacktracking(env)
    if stack:
        env = FrameStack(env, 4)
    return env


def make_env_0_test():
    return make_env(idx=0,myReward=False)
    
def make_env_0():
    return make_env(idx=0,myReward=True)
def make_env_1():
    return make_env(idx=1,myReward=True)
def make_env_2():
    return make_env(idx=2,myReward=True)
def make_env_3():
    return make_env(idx=3,myReward=True)
def make_env_4():
    return make_env(idx=4,myReward=True)
def make_env_5():
    return make_env(idx=5,myReward=True)



class PreprocessFrame(gym.ObservationWrapper):
    """
    Here we do the preprocessing part:
    - Set frame to gray
    - Resize the frame to 96x96x1
    """
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 96
        self.height = 96
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)
        self.level_pred = []
        
    def observation(self, frame):
       
        # Set frame to gray
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Resize the frame to 96x96x1
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = frame[:, :, None]

        return frame
        
class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'], ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): 
        return self._actions[a].copy()

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.

    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.005

class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0
        self.level_pred = env.level_pred

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info


def cal_dist(x1,y1, x2,y2):
    d = ((x1 - x2)**2 + (y1-y2)**2 )**.5
    if x1 >= x2:
        return d
    else:
        return -d

def calcula_deslocamento_por_imagem(img_1, img_2):
    try:
        
        sift = cv2.SIFT_create() #cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img_1,None)
        kp2, des2 = sift.detectAndCompute(img_2,None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)


        # Apply ratio test
        good = []
        for m in matches:
            if m[0].distance < 0.5*m[1].distance: #0.5*m[1].distance:
                good.append(m)
        matches = np.asarray(good)


    
        src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        difs = []
        for i in range(min(len(src), len(dst))):
            difs.append(cal_dist( src[i][0][0], src[i][0][1], dst[i][0][0], dst[i][0][1]))

        difs = np.array(difs)

        dif = np.mean(difs) #int( img_1.shape[1] - np.ceil(np.mean(difs)) )
        #print("img1 shape: {}\nnp.mean(difs): {}\ndif: {}".format(img_1.shape[1],np.mean(difs), dif ) )
        return dif

    except Exception as e: 
        print(e)
        return 0
    
def calcula_deslocamento_por_imagem_fast(img_1, img_2):
    try:
        sift = cv2.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img_1,None)
        kp2, des2 = sift.detectAndCompute(img_2,None)

        #bf = cv2.BFMatcher()
        #matches = bf.knnMatch(des1,des2, k=2)
        
        
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = {} #dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
       



        # Apply ratio test
        good = []
        for m in matches:
            if m[0].distance < 0.5*m[1].distance:
                good.append(m)
        matches = np.asarray(good)

        #print('len of matches {}'.format(len(matches)))
        if len(matches[:,0]) >= 4:
            src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
            dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
            H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        #print H
        else:
            raise AssertionError("Can’t find enough keypoints.")
            #return -1

        difs = []
        for i in range(len(masked)):
            difs.append(cal_dist( src[i][0][0], src[i][0][1], dst[i][0][0], dst[i][0][1]))

        difs = np.array(difs)

        dif = np.mean(difs) #int( img_1.shape[1] - np.ceil(np.mean(difs)) )
        #print("img1 shape: {}\nnp.mean(difs): {}\ndif: {}".format(img_1.shape[1],np.mean(difs), dif ) )
        return dif

    except Exception as e: 
        print(e)
        return 0 
    
class CalcReward(gym.Wrapper):
    """
        my Reward function
    """
    def __init__(self, env):
        super(CalcReward, self).__init__(env)
        
        self.current_image = None
        self.last_image = None
        self.first_timestamp = True
        self.x_max = 0
        self.x_current = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        
        self.current_image = None
        self.last_image = None
        self.first_timestamp = True
        self.x_max = 0
        self.x_current = 0
        return self.env.reset(**kwargs)

    def step(self, action): 
        obs, rew, done, info = self.env.step(action)
        rew = 0
        #print(obs.shape)
        #frame = obs #env.unwrapped.get_screen() #incluir a função de remover o scoreboard? 
        
        
        if self.first_timestamp == True:
            self.last_image = obs
            self.first_timestamp = False
            
        #self.current_image = frame
        
       
        deslocamento_homografia = calcula_deslocamento_por_imagem_fast(self.last_image, obs)
        
        self.x_current += deslocamento_homografia
        
        #print("des_homo:{}\nx_current: {} x_max: {}".format(deslocamento_homografia, self.x_current, self.x_max))
        
        if self.x_current > self.x_max:
            rew = deslocamento_homografia
            self.x_max = self.x_current           
        
        
        self.last_image = obs # atualiza ultimo frame
 
        return obs, rew, done, info
        
    
        
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.level_pred = env.level_pred
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))
       
        
        
class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None
        self.shape = (96, 96, 4)

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

