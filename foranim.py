#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 23:03:52 2020

@author: delis
"""


fig = plt.figure(figsize=(8, 8),facecolor='white')
gs = gridspec.GridSpec(2, 1)

ax2 = plt.subplot(gs[0,0])
quad1 = ax2.pcolormesh(X, Y, z, shading='gouraud')
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$y$')
cb2 = fig.colorbar(quad1,ax=ax2)

ax3 = plt.subplot(gs[1,0])
quad2 = ax3.pcolormesh(X, Y, z, shading='gouraud')
ax3.set_xlabel(r'$x$')
ax3.set_ylabel(r'$y$')
cb3 = fig.colorbar(quad2,ax=ax3)

def init():
    quad1.set_array([])
    quad2.set_array([])
    return quad1, quad2

def animate(iter, array=np.full((N, N), 5)):
    GP.time_evolution_animation(iter)
    quad1.set_array(np.abs(np.conjugate(GP.psi_x)*GP.psi_x).ravel())
    quad2.set_array(np.angle(GP.psi_x).ravel())
    return quad1, quad2

gs.tight_layout(fig)
myarray = np.full((N, N), 5)
anim = animation.FuncAnimation(fig, animate, frames=50,interval=50, blit=False, repeat=False)
FFwriter = animation.FFMpegWriter(fps=5, extra_args=['-vcodec', 'libx264'])
anim.save('lila.mp4', writer=FFwriter)
plt.show()

print('Finished!!')