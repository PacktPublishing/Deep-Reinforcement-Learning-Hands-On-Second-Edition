#!/usr/bin/env python3
import gym
import pybullet_envs

ENV_ID = "MinitaurBulletEnv-v0"
RENDER = True

"""
To fix the error:
libGL error: MESA-LOADER: failed to open iris: /usr/lib/dri/iris_dri.so: 
    cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:$${ORIGIN}/dri:/usr/lib/dri, suffix _dri) libGL error: 
    failed to load driver: iris libGL error: MESA-LOADER: failed to open iris:
    ...
    
use (this is an issue with the underlying library in conda:

$ cd /home/$USER/anaconda3/envs/$ENV/lib
$ mkdir backup  # Create a new folder to keep the original libstdc++
$ mv libstd* backup  # Put all libstdc++ files into the folder, including soft links
$ cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6  ./ # Copy the c++ dynamic link library of the system here
$ ln -s libstdc++.so.6 libstdc++.so
$ ln -s libstdc++.so.6 libstdc++.so.6.0.29

The solution is taken from here https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris 
"""


if __name__ == "__main__":
    spec = gym.envs.registry.spec(ENV_ID)
    spec._kwargs['render'] = RENDER
    env = gym.make(ENV_ID)

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print(env)
    print(env.reset())
    input("Press any key to exit\n")
    env.close()
