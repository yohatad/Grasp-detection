import roboticstoolbox as rtb
import swift
import numpy as np
import spatialmath as sm
import spatialgeometry as sg
from spatialmath import *

env = swift.Swift()
env.launch(realtime=True)

# Intialise the model
puma = rtb.models.Panda()
puma.q = puma.qr

env.add(puma)

Tep = SE3(0.318, -0.2, 0.08435)

#sol = puma.ikine_LM(T)

#Tep = puma.fkine(puma.q)*sm.SE3.Tx(0.2) * SE3.Ty(0.2) * sm.SE3.Tz(0.35)

print(Tep)

axes = sg.Axes(length=1, base = Tep)
env.add(axes)

arrived = False

dt = 0.01

while not arrived:
    v, arrived = rtb.p_servo(puma.fkine(puma.q), Tep, gain = 1, threshold=0.01)
    
    J = puma.jacobe(puma.qr)

    puma.qd = np.linalg.pinv(J) @ v

    env.step(dt)


env.hold()

#puma.qd = [0.1, 0.9, 0.1, 0.1, 0.1, 0.1]

# Add puma to swift

#Tep = puma.fkine([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
