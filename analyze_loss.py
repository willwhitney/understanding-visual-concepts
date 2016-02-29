import os
import pprint
import heapq as H

lastvals = []
bestvals = []
lastvalsdict = {}
bestvalsdict = {}
k =10
suppress_errors = True
tag = 'ballsreg'

for exp in os.listdir('.'):
    if '.py' not in exp and tag in exp:
        lines = open(os.path.join(exp,'val_loss.txt'), 'r').readlines()
        best = (float('inf'),0)
        for i in xrange(len(lines)):
            val_loss = float(lines[i])
            if val_loss < best[0]:
                best = (val_loss, i)
        H.heappush(lastvals, (val_loss,exp))       
        H.heappush(bestvals, (best[0],(exp,best[1])))
        lastvalsdict[exp] = val_loss
        bestvalsdict[exp] = best        
 
def orderbest():
    global bestvals,lastvalsdict
    print '\nbest', k, 'vals'
    nbestvals = H.nsmallest(k,bestvals)
    for pair in nbestvals:
        bestval, info = pair
        exp, epcnum = info
        print exp,'\tbestval',bestval,'at valtest',epcnum,'\tlastval',lastvalsdict[exp]

def orderlast():
    global lastvals,bestvalsdict
    print '\nlast', k, 'vals'
    nlastvals = H.nsmallest(k,lastvals)
    for pair in nlastvals:
        lastval, exp = pair
        bestval, epcnum = bestvalsdict[exp]
        print exp,'\tbestval',bestval,'at valtest',epcnum,'\tlastval',lastval

def orderall():
    """ Find intersection of bestvals and lastvals """
    global bestvals, lastvals,bestvalsdict,lastvalsdict
    print '\nbest', k, 'vals overall'
    nbestvals = H.nsmallest(k,bestvals)
    nlastvals = H.nsmallest(k,lastvals)
    bestexps = []    

    exps = {}
    for i in xrange(len(nlastvals)):
        lastval, explast = nlastvals[i]
        exps[explast] = i
        for j in xrange(len(nbestvals)):
            bestval, info = nbestvals[j]
            expbest, epcnum = info
            if expbest == explast:
                exps[expbest] += j
    for pair in exps.items():
        H.heappush(bestexps,(pair[1],pair[0]))
    nbestexps = H.nsmallest(k,bestexps)
    for pair in nbestexps:
        rank, exp = pair
        lastval = lastvalsdict[exp]
        bestval, epcnum = bestvalsdict[exp]
        print exp,'\tbestval',bestval,'at valtest',epcnum,'\tlastval',lastval  
    

orderbest()
orderlast()
orderall()

