import cCOSMO
import os
import numpy as np
import json
import math



def COSMOSAC_gamma(sigmas,x,T,using2010=True):
    with open("sigmas3.txt","w") as f:
        for id,i in enumerate(sigmas):
            print(str(id)+"      "+i,file=f)
    db = cCOSMO.DelawareProfileDatabase("sigmas3.txt","./")
    for iden in sigmas:
        db.add_profile(db.normalize_identifier(iden))
    COSMO = cCOSMO.COSMO1(sigmas, db)
    os.system("rm -f sigmas3.txt")
    if using2010:
        return np.exp((COSMO.get_lngamma_comb(T, x)+COSMO.get_lngamma_resid(T, x)).tolist()).tolist()
    return np.exp(np.array(COSMO.get_lngamma(T, x).tolist())).tolist()



