import numpy as np 
# Calculate equal area partition of unit sphere with "num_partitions" partitions
def partition_sphere(num_partitions):

    # Functions to obtain angle from cap area and viceversa
    def angle_to_cap_area(phi):
        return 4*np.pi*(np.sin(phi/2.0))**2

    def cap_area_to_angle(area):
        return 2.0*np.arcsin(np.sqrt(area/(4.0*np.pi)))

    # Calculate ares of each state and polar caps angle (phi0 and pi-phi0)
    state_area = 4*np.pi/num_partitions
    phi0 = cap_area_to_angle(state_area)
    # Calculate the number of collars between the polar caps
    ideal_collar_angle = np.sqrt(state_area)
    ideal_num_collars = (np.pi - 2*phi0)/ideal_collar_angle
    num_collars = int(max(1,np.round(ideal_num_collars)))
    if num_partitions == 2:
        num_collars = 0
    collar_angle = (np.pi - 2*phi0)/num_collars
    # Initialize variables for number of regions in each collar
    ideal_regionsPerCollar = np.zeros(num_collars)
    phis = np.zeros(num_collars + 1)
    regionsPerCollar = np.zeros(num_collars)
    thetas = []
    a = [0]
    # Iterate over each collar to get right number of regions per collar
    # and correct location of phi angles of each collar
    for i in range(num_collars):
        # Calculate num of regions in collar i
        cap_area_phi1 = angle_to_cap_area(phi0 + i*collar_angle)
        cap_area_phi2 = angle_to_cap_area(phi0 + (i+1)*collar_angle)
        ideal_regionsPerCollar[i] = (cap_area_phi2 - cap_area_phi1)/state_area
        regionsPerCollar[i] = np.round(ideal_regionsPerCollar[i] + a[i])
        # Correct values of phi around collar i
        suma = 0
        for j in range(i+1):
            suma = suma + ideal_regionsPerCollar[j] - regionsPerCollar[j]
        a.append(suma)
        summ = 1
        for j in range(i):
            summ = summ + regionsPerCollar[j]
        phis[i] = cap_area_to_angle(summ*state_area)
        phis[-1] = np.pi - phi0
        # Obtain list of thetas for a given collar
        thetasi = np.zeros(int(regionsPerCollar[i]))
        dth = 2.0*np.pi/regionsPerCollar[i]
        for j in range(int(regionsPerCollar[i])):
            thetasi[j] = j*dth
        thetas.append(thetasi)
    phis = np.insert(phis,0,0)
    regionsPerCollar = np.append(regionsPerCollar,1)
    regionsPerCollar = np.insert(regionsPerCollar,0,1)
        # return number of regions for all collars, 
        # phi angles of collars and theta angles for each collar
    return regionsPerCollar.astype(np.int32), phis, thetas
