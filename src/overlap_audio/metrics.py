import torch

def f1_per_class(targets, predictions):
    tp=(targets*predictions).sum(0);fp=((1-targets)*predictions).sum(0);fn=(targets*(1-predictions)).sum(0)
    return 2*tp/(2*tp+fp+fn).clamp_min(1)

def thresholds_from_validation(targets, probabilities):
    thresholds=[]
    for c in range(targets.shape[1]):
        if targets[:,c].sum()==0: thresholds.append(.5);continue
        grid=torch.linspace(.1,.9,17)
        scores=[float(f1_per_class(targets[:,c:c+1],(probabilities[:,c:c+1]>=t).float())[0]) for t in grid]
        thresholds.append(float(grid[max(range(len(scores)),key=lambda i:scores[i])]))
    return torch.tensor(thresholds)

def average_precision(target, scores):
    positives=float(target.sum())
    if not positives: return None
    # Group tied scores, using precision after every complete tied threshold.
    total=correct=0.;ap=0.
    for score in torch.unique(scores,sorted=True).flip(0):
        selected=scores==score;n=float(selected.sum());hits=float(target[selected].sum())
        total+=n;correct+=hits;ap+=(hits/positives)*(correct/total)
    return min(1.,max(0.,ap))

def evaluate(targets, probabilities, thresholds):
    predictions=(probabilities>=thresholds).float()
    tp=float((targets*predictions).sum());fp=float(((1-targets)*predictions).sum());fn=float((targets*(1-predictions)).sum())
    aps=[average_precision(targets[:,c],probabilities[:,c]) for c in range(targets.shape[1])]
    supported=[a for a in aps if a is not None]
    return {"macro_f1":float(f1_per_class(targets,predictions).mean()),"micro_f1":2*tp/max(1.,2*tp+fp+fn),"per_class_f1":f1_per_class(targets,predictions).tolist(),"per_class_average_precision":aps,"macro_average_precision":sum(supported)/len(supported) if supported else None,"positive_support":targets.sum(0).int().tolist(),"samples":len(targets)}
