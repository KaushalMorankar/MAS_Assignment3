import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx

N=20
p=0.3
np.random.seed(42)

G=nx.erdos_renyi_graph(N,p,seed=42)
while not nx.is_connected(G):
    G=nx.erdos_renyi_graph(N,p,seed=np.random.randint(1000))

L=nx.laplacian_matrix(G).toarray()

t_k=np.array([
    [1.0,1.0],[1.0,2.0],[1.0,3.0],[1.0,4.0],[1.0,5.0],
    [1.0,1.5],[1.0,4.5],[1.5,3.0],[2.0,3.6],[2.5,4.2],[3.0,4.8],
    [2.0,2.4],[2.5,1.8],[3.0,1.2],
    [1.25,3.0],[1.75,3.3],[2.25,3.9],[1.75,2.7],[2.25,2.1],[2.75,1.5]
])

t_a=np.array([
    [1.0,1.0],[1.3,2.0],[1.6,3.0],[1.9,4.0],[2.2,5.0],
    [2.5,4.0],[2.8,3.0],[3.1,2.0],[3.4,1.0],
    [1.45,2.5],[1.75,2.5],[2.05,2.5],[2.35,2.5],[2.65,2.5],[2.95,2.5],
    [1.15,1.5],[1.45,3.5],[2.35,4.5],[2.65,3.5],[2.95,1.5]
])

t_u=np.array([
    [1.0,5.0],[1.0,4.0],[1.0,3.0],[1.0,2.0],[1.1,1.4],[1.4,1.0],[1.9,1.0],
    [2.4,1.0],[2.9,1.0],[3.2,1.4],[3.3,2.0],[3.3,3.0],[3.3,4.0],[3.3,5.0],
    [1.0,4.5],[1.0,2.5],[1.65,1.0],[2.65,1.0],[3.3,2.5],[3.3,4.5]
])

t_s=np.array([
    [3.2,4.5],[2.8,5.0],[2.2,5.0],[1.6,4.8],[1.2,4.2],[1.2,3.6],[1.6,3.1],
    [2.2,2.9],[2.8,2.7],[3.2,2.2],[3.2,1.6],[2.8,1.1],[2.2,1.0],[1.5,1.0],[1.0,1.3],
    [2.5,5.0],[1.4,4.5],[2.5,2.8],[3.0,1.9],[1.8,1.0]
])

t_h=np.array([
    [1.0,1.0],[1.0,2.0],[1.0,3.0],[1.0,4.0],[1.0,5.0],
    [3.3,1.0],[3.3,2.0],[3.3,3.0],[3.3,4.0],[3.3,5.0],
    [1.4,3.0],[1.8,3.0],[2.15,3.0],[2.5,3.0],[2.9,3.0],
    [1.0,1.5],[1.0,4.5],[3.3,1.5],[3.3,4.5],[2.15,3.0]
])

t_l=np.array([
    [1.0,5.0],[1.0,4.2],[1.0,3.4],[1.0,2.6],[1.0,1.8],[1.0,1.0],
    [1.4,1.0],[1.8,1.0],[2.2,1.0],[2.6,1.0],[3.0,1.0],[3.4,1.0],
    [1.0,4.6],[1.0,3.8],[1.0,3.0],[1.0,2.2],[1.0,1.4],
    [1.6,1.0],[2.4,1.0],[3.2,1.0]
])

letters={
    'K':t_k,
    'A1':t_a,
    'U':t_u,
    'S':t_s,
    'H':t_h,
    'A2':t_a,
    'L':t_l
}
seq_key=['K','A1','U','S','H','A2','L']
seq_name=['K','A','U','S','H','A','L']

x=np.random.uniform(0.0,4.0,(N,2))
trajectory=[x.copy()]
letter_labels=["Start"]
errnorm=[]

dt=0.05
spl=80

for key,name in zip(seq_key,seq_name):
    g=letters[key]
    for step in range(spl):
        dx=-L@(x-g)
        x+=dt*dx
        trajectory.append(x.copy())
        letter_labels.append(f"Target Formation:Letter {name}")
        err=np.linalg.norm((x-np.mean(x,axis=0))-(g-np.mean(g,axis=0)))
        errnorm.append(err)

trajectory=np.array(trajectory)

fig_seq,axes=plt.subplots(2,4,figsize=(14,7))
axes=axes.flatten()

for idx,(key,name) in enumerate(zip(seq_key,seq_name)):
    step_idx=(idx+1)*spl
    ax=axes[idx]
    pos=trajectory[step_idx]
    ax.scatter(pos[:,0],pos[:,1],c='blue',s=45,zorder=3)
    for u,v in G.edges():
        ax.plot([pos[u,0],pos[v,0]],[pos[u,1],pos[v,1]],'gray',alpha=0.5,zorder=1)
    ax.set_title(f'Letter {name}',fontsize=12,fontweight='bold')
    ax.set_xlim(0.0,4.4)
    ax.set_ylim(0.0,6.0)
    ax.grid(True,linestyle='--',alpha=0.5)

axes[7].axis('off') 
plt.tight_layout()
plt.savefig('p1_formation_sequence.png',dpi=300)
plt.close(fig_seq)

fig_err,ax_err=plt.subplots(figsize=(8,4))
ax_err.plot(errnorm,'r-',linewidth=1.8)
for i in range(1,len(seq_key)):
    ax_err.axvline(i*spl,color='gray',linestyle='--',alpha=0.7)
ax_err.set_title('Formation Deviation $\|x-g-\\bar{x}\|_F$ Across KAUSHAL Sequence',fontsize=12,fontweight='bold')
ax_err.set_xlabel('Simulation Step')
ax_err.set_ylabel('Formation Deviation')
ax_err.grid(True,linestyle=':',alpha=0.6)
plt.tight_layout()
plt.savefig('p1_convergence.png',dpi=300)
plt.close(fig_err)

fig,ax=plt.subplots(figsize=(7,7))

scatter=ax.scatter([],[],c='blue',s=70,zorder=3,label='Agents ($N=20$)')
lines=[ax.plot([],[],'gray',alpha=0.4,zorder=1)[0] for _ in G.edges()]
title_text=ax.set_title('',fontsize=13,fontweight='bold')

ax.set_xlim(0.0,4.4)
ax.set_ylim(0.0,6.0)
ax.set_xlabel('X coordinate',fontsize=11)
ax.set_ylabel('Y coordinate',fontsize=11)
ax.grid(True,linestyle='--',alpha=0.5)
ax.legend(loc='upper right')

def init():
    scatter.set_offsets(np.empty((0,2)))
    for line in lines:
        line.set_data([],[])
    title_text.set_text('')
    return [scatter,title_text]+lines

def update(frame):
    pos=trajectory[frame]
    scatter.set_offsets(pos)
    for idx,(u,v) in enumerate(G.edges()):
        lines[idx].set_data([pos[u,0],pos[v,0]],[pos[u,1],pos[v,1]])
    title_text.set_text(f'{letter_labels[frame]} (Step {frame}/{len(trajectory)-1})')
    return [scatter,title_text]+lines

anim=animation.FuncAnimation(fig,update,frames=len(trajectory),init_func=init,interval=40,blit=True)

output_gif='kaushal.gif'
anim.save(output_gif,writer='pillow',fps=25)
print(f'Animation video successfully generated and saved to {output_gif}')

plt.close()
