"""Shared corrected outer-pair seed contract and independent coordinate audit."""
import numpy as np

CONFIGS=(('h8',.125),('h4',.25))
LEG_FACTORS=np.array((-.125,.125,-.0625,.0625))

def make_seeds(midpoint,width):
    midpoint=np.asarray(midpoint,float);width=np.asarray(width,float)
    assert midpoint.shape==(3,) and width.shape==(3,) and np.all(width>0)
    seeds=[];metadata=[]
    for label,alpha in CONFIGS:
        for axis in range(3):
            for side in (-1,1):
                p=midpoint.copy();p[axis]+=side*alpha*width[axis]/2
                seeds.append(p)
                metadata.append(dict(config=label,alpha=alpha,axis=axis,side=side,
                                     physical_width=float(width[axis]),half_offset=float(alpha*width[axis]/2),pair_separation=float(alpha*width[axis])))
    return np.asarray(seeds),metadata

def audit_coordinates(midpoint,width,seeds,metadata,atol=2e-14):
    midpoint=np.asarray(midpoint,float);width=np.asarray(width,float);seeds=np.asarray(seeds,float)
    assert seeds.shape==(12,3) and len(metadata)==12
    defects=[]
    for i,entry in enumerate(metadata):
        axis=int(entry['axis']);side=int(entry['side']);alpha=float(entry['alpha'])
        expected=midpoint.copy();expected[axis]+=side*alpha*width[axis]/2
        defects.append(np.max(abs(seeds[i]-expected))/width[axis])
        defects.append(abs((seeds[i,axis]-midpoint[axis])/width[axis]-side*alpha/2))
    for conf,(label,alpha) in enumerate(CONFIGS):
        for axis in range(3):
            neg,pos=6*conf+2*axis,6*conf+2*axis+1
            assert metadata[neg]['config']==metadata[pos]['config']==label
            defects.append(abs((seeds[pos,axis]-seeds[neg,axis])/width[axis]-alpha))
            defects.append(np.max(abs((seeds[pos]+seeds[neg])/2-midpoint))/width[axis])
            for j in range(3):
                if j!=axis: defects.append(max(abs(seeds[pos,j]-midpoint[j]),abs(seeds[neg,j]-midpoint[j]))/width[j])
    max_defect=float(max(defects))
    assert max_defect<atol,(max_defect,atol)
    return max_defect

def regression_check():
    # This counterexample is the previous bug: alpha .25/.5 labels h8/h4.
    q=np.array((.984375,1.3,5.9));w=np.array((.03125,.19634954084936207,.19634954084936207))
    seeds,meta=make_seeds(q,w);good=audit_coordinates(q,w,seeds,meta)
    wrong=seeds.copy()
    for i,m in enumerate(meta):
        axis=m['axis'];wrong[i,axis]=q[axis]+m['side']*(2*m['alpha'])*w[axis]/2
    caught=False
    try:audit_coordinates(q,w,wrong,meta)
    except AssertionError:caught=True
    assert caught,'factor-of-two seed displacement escaped coordinate audit'
    return dict(correct_max_normalized_defect=good,prior_factor_two_caught=caught)
