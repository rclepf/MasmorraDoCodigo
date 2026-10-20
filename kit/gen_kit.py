import numpy as np, math, os, zipfile
from PIL import Image, ImageDraw
import random
OUT='kit/Dungeon3D_Kit'; os.makedirs(OUT+'/Models',exist_ok=True); os.makedirs(OUT+'/Textures',exist_ok=True)
MATS={'stone':(.45,.47,.52),'stone_dark':(.28,.30,.35),'wood':(.45,.28,.14),'metal':(.62,.64,.68),'gold':(.95,.72,.15),
 'flame':(1,.55,.1),'player':(.2,.45,.9),'slime':(.35,.8,.3),'bone':(.9,.88,.8),'potion':(.85,.1,.2),'eye':(.05,.05,.05)}
class M:
    def __init__(s): s.v=[];s.vt=[];s.vn=[];s.f=[]  # f: (mat, [(vi,ti,ni)*])
    def _add(s,mat,verts,uvs,n,quad=True):
        P=[np.array(x) for x in verts]
        if np.dot(np.cross(P[1]-P[0],P[2]-P[0]),n)<0: verts=verts[::-1];uvs=uvs[::-1]
        b=len(s.v);s.v+=verts;nb=len(s.vn);s.vn.append(n);tb=len(s.vt);s.vt+=uvs
        idx=[(b+i+1,tb+i+1,nb+1) for i in range(len(verts))]
        s.f.append((mat,idx))
    def box(s,mat,c,sz,rot=0):
        cx,cy,cz=c;sx,sy,sz_=[a/2 for a in sz]
        P=lambda x,y,z:(cx+x,cy+y,cz+z)
        faces=[((0,0,1),[(-sx,-sy,sz_),(sx,-sy,sz_),(sx,sy,sz_),(-sx,sy,sz_)]),
               ((0,0,-1),[(sx,-sy,-sz_),(-sx,-sy,-sz_),(-sx,sy,-sz_),(sx,sy,-sz_)]),
               ((1,0,0),[(sx,-sy,sz_),(sx,-sy,-sz_),(sx,sy,-sz_),(sx,sy,sz_)]),
               ((-1,0,0),[(-sx,-sy,-sz_),(-sx,-sy,sz_),(-sx,sy,sz_),(-sx,sy,-sz_)]),
               ((0,1,0),[(-sx,sy,sz_),(sx,sy,sz_),(sx,sy,-sz_),(-sx,sy,-sz_)]),
               ((0,-1,0),[(-sx,-sy,-sz_),(sx,-sy,-sz_),(sx,-sy,sz_),(-sx,-sy,sz_)])]
        for n,vs in faces: s._add(mat,[P(*v) for v in vs],[(0,0),(1,0),(1,1),(0,1)],n)
    def cyl(s,mat,c,r,h,seg=16):
        cx,cy,cz=c
        for i in range(seg):
            a0,a1=2*math.pi*i/seg,2*math.pi*(i+1)/seg;am=(a0+a1)/2
            p=lambda a,y:(cx+r*math.cos(a),cy+y,cz+r*math.sin(a))
            s._add(mat,[p(a0,-h/2),p(a1,-h/2),p(a1,h/2),p(a0,h/2)],[(i/seg,0),((i+1)/seg,0),((i+1)/seg,1),(i/seg,1)],(math.cos(am),0,math.sin(am)))
            s._add(mat,[(cx,cy+h/2,cz),p(a0,h/2),p(a1,h/2)][::-1],[(.5,.5),(0,0),(1,0)][::-1],(0,1,0))
            s._add(mat,[(cx,cy-h/2,cz),p(a1,-h/2),p(a0,-h/2)][::-1],[(.5,.5),(0,0),(1,0)][::-1],(0,-1,0))
    def sphere(s,mat,c,r,sy=1,seg=16,ring=10):
        cx,cy,cz=c
        pt=lambda i,j:(cx+r*math.sin(math.pi*j/ring)*math.cos(2*math.pi*i/seg),cy+sy*r*math.cos(math.pi*j/ring),cz+r*math.sin(math.pi*j/ring)*math.sin(2*math.pi*i/seg))
        for j in range(ring):
            for i in range(seg):
                vs=[pt(i,j),pt(i+1,j),pt(i+1,j+1),pt(i,j+1)]
                n=np.array([(vs[0][k]+vs[2][k])/2-c[k] for k in range(3)]);n=tuple(n/np.linalg.norm(n))
                # winding: outward
                s._add(mat,vs[::-1],[(0,0),(1,0),(1,1),(0,1)],n)
    def save(s,name):
        used=sorted({m for m,_ in s.f})
        with open(f'{OUT}/Models/{name}.mtl','w') as f:
            for m in used: f.write(f'newmtl {m}\nKd {MATS[m][0]} {MATS[m][1]} {MATS[m][2]}\nKa 1 1 1\n'+('Ke 1 .5 .1\n' if m=='flame' else '')+'\n')
        with open(f'{OUT}/Models/{name}.obj','w') as f:
            f.write(f'mtllib {name}.mtl\no {name}\n')
            for v in s.v: f.write('v %.4f %.4f %.4f\n'%v)
            for t in s.vt: f.write('vt %.4f %.4f\n'%t)
            for n in s.vn: f.write('vn %.4f %.4f %.4f\n'%tuple(n))
            cur=None
            for m,idx in s.f:
                if m!=cur: f.write(f'usemtl {m}\n');cur=m
                f.write('f '+' '.join(f'{a}/{b}/{c}' for a,b,c in idx)+'\n')
        return len(s.f)
def mk(name,fn):
    m=M();fn(m);return name,m.save(name)
def floor(m): m.box('stone',(0,-.1,0),(4,.2,4))
def wall(m): m.box('stone',(0,1.5,0),(4,3,.4)); m.box('stone_dark',(0,3.05,0),(4.1,.1,.5))
def walldoor(m):
    m.box('stone',(-1.5,1.5,0),(1,3,.4));m.box('stone',(1.5,1.5,0),(1,3,.4));m.box('stone',(0,2.6,0),(2,.8,.4));m.box('stone_dark',(0,3.05,0),(4.1,.1,.5))
def door(m): m.box('wood',(0,1.1,0),(1.9,2.2,.15));m.box('metal',(.6,1.1,.1),(.12,.12,.08));m.box('metal',(0,.5,.09),(1.9,.1,.02));m.box('metal',(0,1.7,.09),(1.9,.1,.02))
def pillar(m): m.cyl('stone',(0,1.5,0),.35,3);m.cyl('stone_dark',(0,.1,0),.5,.2);m.cyl('stone_dark',(0,2.9,0),.5,.2)
def crate(m): m.box('wood',(0,.5,0),(1,1,1));[m.box('wood',(0,.5,0),(1.04,.12,1.04)) for _ in (0,)]
def barrel(m): m.cyl('wood',(0,.6,0),.4,1.2);m.cyl('metal',(0,.3,0),.42,.1);m.cyl('metal',(0,.9,0),.42,.1)
def torch(m): m.box('wood',(0,.6,0),(.1,1.2,.1));m.sphere('flame',(0,1.35,0),.18,1.4)
def chest(m): m.box('wood',(0,.3,0),(1,.6,.6));m.box('wood',(0,.72,0),(1,.24,.6));m.box('metal',(0,.5,.31),(.14,.2,.04));m.box('metal',(-.4,.4,0),(.08,.8,.62));m.box('metal',(.4,.4,0),(.08,.8,.62))
def coin(m): m.cyl('gold',(0,.5,0),.25,.05,20)
def key(m): m.cyl('gold',(0,.5,0),.12,.04,12);m.box('gold',(0,.5,-.3),(.05,.04,.5));m.box('gold',(.06,.5,-.5),(.12,.04,.06))
def potion(m): m.sphere('potion',(0,.25,0),.2);m.cyl('metal',(0,.5,0),.07,.2,10);m.box('wood',(0,.63,0),(.1,.08,.1))
def sword(m): m.box('metal',(0,.9,0),(.1,1.1,.03));m.box('gold',(0,.3,0),(.4,.06,.08));m.box('wood',(0,.12,0),(.08,.3,.08))
def player(m): m.cyl('player',(0,.8,0),.35,1.0,16);m.sphere('player',(0,1.3,0),.35);m.sphere('player',(0,.3,0),.35);m.sphere('bone',(0,1.75,0),.28);m.box('eye',(-.1,1.78,.25),(.06,.06,.04));m.box('eye',(.1,1.78,.25),(.06,.06,.04))
def slime(m): m.sphere('slime',(0,.4,0),.5,.8);m.sphere('eye',(-.17,.55,.4),.06);m.sphere('eye',(.17,.55,.4),.06)
def skeleton(m): m.box('bone',(0,1.0,0),(.5,.7,.25));m.sphere('bone',(0,1.65,0),.25);m.box('bone',(-.15,.3,0),(.12,.6,.12));m.box('bone',(.15,.3,0),(.12,.6,.12));m.box('bone',(-.38,1.0,0),(.1,.6,.1));m.box('bone',(.38,1.0,0),(.1,.6,.1));m.box('eye',(-.09,1.68,.22),(.07,.07,.04));m.box('eye',(.09,1.68,.22),(.07,.07,.04))
names=[('Floor_Tile',floor),('Wall',wall),('Wall_Door',walldoor),('Door',door),('Pillar',pillar),('Crate',crate),('Barrel',barrel),('Torch',torch),('Chest',chest),('Coin',coin),('Key',key),('Potion',potion),('Sword',sword),('Player',player),('Enemy_Slime',slime),('Enemy_Skeleton',skeleton)]
info=[mk(n,f) for n,f in names]
# texture tileável de pedra para prática de UV/materiais
random.seed(3);W=256;img=Image.new('RGB',(W,W),(110,114,124));d=ImageDraw.Draw(img)
for row in range(8):
    off=(row%2)*16
    for col in range(-1,9):
        x=col*32+off;y=row*32;g=random.randint(-14,14);c=(112+g,116+g,128+g)
        d.rectangle([x+1,y+1,x+30,y+30],fill=c,outline=(70,72,80))
        for _ in range(12): px=x+random.randint(2,29);py=y+random.randint(2,29);d.point((px,py),fill=(c[0]-20,c[1]-20,c[2]-20))
img.save(OUT+'/Textures/Stone_Albedo.png')
open(OUT+'/LEIA-ME.txt','w',encoding='utf-8').write('''KIT DUNGEON 3D - Codificação de Sistemas de Jogos Digitais
Escala: 1 unidade = 1 metro. Tile de piso/parede = 4 x 4 m. Pivô na base, no centro do modelo.
Formato: OBJ + MTL (a Unity importa nativamente; cada cor vira um material).
Como usar: copie a pasta Models e Textures para Assets/ do projeto Unity.
Material 'flame' tem emissão no MTL - ajuste Emission no Material, se necessário.
Os modelos são primitivas simples, feitos por código, para as aulas funcionarem desde o início.
Substitua-os por modelos feitos no Blender (baú, galpão, props dos alunos) ou por pacotes CC0 (Kenney, Quaternius).
Textura Stone_Albedo.png (256x256, tileável) para praticar mapeamento e materiais.
Licença: CC0 (domínio público).
''')
print(info)
z=zipfile.ZipFile('kit/Dungeon3D_Kit.zip','w',zipfile.ZIP_DEFLATED)
for r,_,fs in os.walk(OUT):
    for f in fs: p=os.path.join(r,f);z.write(p,os.path.relpath(p,'kit'))
z.close()
