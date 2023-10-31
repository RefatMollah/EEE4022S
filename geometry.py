import numpy as np

def unitVector(vector):
     """returns unit vector"""
     return vector / np.linalg.norm(vector)

def calculate_Angle(v1,v2):
    """returns angle between two vectors v1 and v2"""
    unit1 = unitVector(v1)
    unit2 = unitVector(v2)
    return np.arccos(np.clip(np.dot(unit1, unit2), -1.0, 1.0))

def calculate_scalarProjection(v1,v2):
    """returns the scalar projection of vector v1 onto v2"""
    return np.dot(v1,v2) / np.linalg.norm(v2)

def monostatic_delay(c,radar,Target):
    """
    Args:
        c (_type_): _description_
        radar (_type_): _description_
        Target (_type_): _description_
    """
    distance = np.subtract(Target,radar)
    #magnitude of distance vector
    r = np.linalg.norm(distance)
    return 2 * r / c

def monstatic_doppler(c ,fc ,radar, target):
    """_summary_

    Args:
        c (_type_): _description_
        fc (_type_): _description_
        radarPos (_type_): _description_
        target (_type_): _description_
    """
    #Determine radar-target line-of-sight
    los = target.coordinates - radar.coordinates
    #project velocity onto los
    radialVelocity = np.dot(target.velocity,los)/np.linalg.norm(los)
    #compute doppler shift
    fd = 2 * radialVelocity * fc / c

def bistatic_delay(c,Rx,Tx,Target):
    """_summary_

    Args:
        Rx (_type_): _description_
        Tx (_type_): _description_
        Target (_type_): _description_
    """
    R_rx = np.subtract(Target.coordinates, Rx.coordinates)
    R_tx = np.subtract(Target.coordinates, Tx.coordinates)
    
    return (np.linalg.norm(R_tx)/c) + (np.linalg.norm(R_rx)/c)

def bistatic_TargetRange(Tx,Rx,Target):
    """_summary_

    Args:
        Rx (_type_): _description_
        Tx (_type_): _description_
        Target (_type_): _description_
    """
    R_rx = np.subtract(Target.coordinates, Rx.coordinates)
    R_tx = np.subtract(Target.coordinates, Tx.coordinates)
    
    return (np.linalg.norm(R_tx)) + (np.linalg.norm(R_rx))


def calculate_TargetRange(Radar,Target):
    """_summary_

    Args:
        Rx (_type_): _description_
        Tx (_type_): _description_
        Target (_type_): _description_
    """
    R_rx = np.subtract(Target.coordinates, Radar.coordinates)

    return np.linalg.norm(R_rx)

def calculate_Bisector(v1,v2):
    """_summary_

    Args:
        v1 (_type_): _description_
        v2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    normV1 = v1/np.linalg.norm(v1)
    normV2 = v2/np.linalg.norm(v2)
    return np.add(normV1,normV2)

def calculate_DopplerShift(TxPos,RxPos,Target,fc,c):
    """_summary_

    Args:
        Tx (_type_): _description_
        Rx (_type_): _description_
        Target (_type_): _description_
        fc (_type_): _description_
        c (_type_): _description_

    Returns:
        _type_: _description_
    """
    #determine TX-target-Rx vectors
    v1 = np.subtract(Target.coordinates,TxPos)
    v2 = np.subtract(Target.coordinates,RxPos)
    #compute bistatic vector and angle
    bistaticVector = calculate_Bisector(v1,v2)
    bistaticAngle = calculate_Angle(v1,v2)
    #calculate radial velocity
    radialVelocity = calculate_scalarProjection(Target.velocity,bistaticVector)*np.cos(bistaticAngle/2)
    #determine doppler shift
    return 2 * radialVelocity * fc / c

def calculate_DopplerShift2(TxPos,RxPos,Target,fc,c):
    v1 = np.subtract(Target.coordinates,TxPos)
    v2 = np.subtract(Target.coordinates,RxPos)
    
    rxVelocity = calculate_scalarProjection(Target.velocity,v1)
    txVelocity = calculate_scalarProjection(Target.velocity,v2)
    
    return (fc/c) * (rxVelocity + txVelocity)
    

def calculate_RadarRange(peakPower, transmitterGain, receiverGain, wavelength, rcs, transmitterRange, receiverRange):
    """_summary_

    Args:
        peakPower (_type_): _description_
        transmittergain (_type_): _description_
        receiverGain (_type_): _description_
        wavelength (_type_): _description_
        rcs (_type_): _description_
        transmitterRange (_type_): _description_
        receiverRange (_type_): _description_
    """
    P = ( peakPower * transmitterGain * receiverGain * wavelength**2 * rcs ) / ((64* np.pi**2) * (transmitterRange**2) * (receiverRange**2))
    
    return P
    
    