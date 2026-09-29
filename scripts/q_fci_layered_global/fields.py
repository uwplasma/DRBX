"""Q's retained field catalogue with analytic field-derived normal data.

Values and logical gradients only; physical normals belong to the geometry.
No geometry-dependent correction is included in any field.
"""
import numpy as np

WAVES = tuple((f'{direction}_lambda{lam:g}_m1', direction, lam)
              for direction in ('x', 'y') for lam in (.5, .25, 2., 4.))


def common(points):
    p = np.asarray(points, float).reshape(-1, 3)
    u, theta, eta = p.T
    x, y = u * np.cos(theta), u * np.sin(theta)
    v = 1 + .1*x*np.cos(eta) + .05*y*np.sin(eta) + .02*(x*x-y*y)*np.cos(2*eta)
    fx = .1*np.cos(eta) + .04*x*np.cos(2*eta)
    fy = .05*np.sin(eta) - .04*y*np.cos(2*eta)
    fe = -.1*x*np.sin(eta) + .05*y*np.cos(eta) - .04*(x*x-y*y)*np.sin(2*eta)
    g = np.column_stack((fx*np.cos(theta)+fy*np.sin(theta),
                         u*(-fx*np.sin(theta)+fy*np.cos(theta)), fe))
    return v, g


def evaluate(points, name):
    p = np.asarray(points, float).reshape(-1, 3)
    if name == 'constant':
        return np.ones(len(p)), np.zeros_like(p)
    if name == 'common':
        return common(p)
    if name in ('homogeneous_D', 'simple_zero_N'):
        v, g = common(p)
        u = p[:, 0]
        envelope = (1-u*u)**2
        gradient = envelope[:, None]*g
        gradient[:, 0] += -4*u*(1-u*u)*v
        return envelope*v + (1 if name == 'simple_zero_N' else 0), gradient
    for wave_name, direction, wavelength in WAVES:
        if name != wave_name:
            continue
        u, theta, eta = p.T
        k = 2*np.pi/wavelength
        if direction == 'x':
            z, du, dt = u*np.cos(theta), k*np.cos(theta), -k*u*np.sin(theta)
        else:
            z, du, dt = u*np.sin(theta), k*np.sin(theta), k*u*np.cos(theta)
        value = np.exp(1j*(k*z+eta))
        return value, 1j*value[:, None]*np.column_stack((du, dt, np.ones(len(p))))
    raise ValueError(name)


def callable_field(name):
    return lambda points: evaluate(points, name)

FIELDS=('common','homogeneous_D','x_lambda2_m1','x_lambda4_m1','y_lambda2_m1','y_lambda4_m1','constant','fA','fB')+tuple(f'wave_a{a}_lambda{l}' for a in (30,60,120,150) for l in (2,4))+('simple_zero_N',)
def field(name,p):
    p=np.asarray(p).reshape(-1,3);u,t,e=p.T;x=u*np.cos(t);y=u*np.sin(t)
    if name.startswith('wave_a'):
        angle,lam=name.removeprefix('wave_a').split('_lambda');a=np.deg2rad(float(angle));k=2*np.pi/float(lam);c=np.cos(t-a);s=np.sin(t-a);v=np.exp(1j*(k*u*c+e))
        return v,1j*v[:,None]*np.column_stack((k*c,-k*u*s,np.ones(len(p))))
    if name not in ('fA','fB'):return evaluate(p,name)
    if name=='fA':
        a=1+.1*x+.07*y;z=3*e+.37;v=a*np.cos(z);fx=.1*np.cos(z);fy=.07*np.cos(z);fe=-3*a*np.sin(z)
    else:
        a=np.exp(.2*x-.15*y);z=5*e+.61;v=a*np.sin(z);fx=.2*v;fy=-.15*v;fe=5*a*np.cos(z)
    return v,np.column_stack((fx*np.cos(t)+fy*np.sin(t),u*(-fx*np.sin(t)+fy*np.cos(t)),fe))
def all_fields(p):
    z=[field(f,p) for f in FIELDS];return np.stack([x[0] for x in z],axis=-1),np.stack([x[1] for x in z],axis=-1)
