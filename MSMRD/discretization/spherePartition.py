import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Calculate equal area partition of unit sphere with "num_partitions" partitions
def partitionSphere(num_partitions):

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

    # Plot spherical partition in 3D
def plotPartitionedSphere(numRegionsCollar = None, phis = None, thetas = None, save = None):
    if numRegionsCollar == None:
        numRegionsCollar, phis, thetas = partitionSphere(numPartitions)
    if save == None:
        save = False
    fig = plt.figure(figsize=plt.figaspect(0.95)*1.5)
    ax = fig.gca(projection='3d')
    ax._axis3don = False
    r=1
    # For porper viewing with mplot3d
    minth = 0 
    maxth = np.pi

    # Plot inner white sphere
    u = np.linspace(0, 2 * np.pi, 400)
    v = np.linspace(0, np.pi, 400)
    xx = r * np.outer(np.cos(u), np.sin(v))
    yy = r * np.outer(np.sin(u), np.sin(v))
    zz = r * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xx, yy, zz, color='white', linewidth=0, antialiased=False, alpha = 1.0)

    # Plot collars
    for phi in phis:
        theta = np.linspace(minth,maxth, 50)
        x = r * np.sin(theta) * np.sin(phi)
        y = r * np.cos(theta) * np.sin(phi)
        z = r * np.cos(phi) 
        ax.plot(x, y, z, '-k')
    # Plot divisions in every collar
    for i in range(len(numRegionsCollar)):
        numDiv = int(numRegionsCollar[i])
        if numDiv > 1:
            dth = 2 * np.pi /numDiv
            phi = np.linspace(phis[i],phis[i+1],10)
            for j in range(numDiv):
                theta = j*dth
                if (theta >= minth) and (theta <= maxth):
                    x = r * np.sin(theta) * np.sin(phi)
                    y = r * np.cos(theta) * np.sin(phi)
                    z = r * np.cos(phi)
                    ax.plot(x, y, z, '-k')

    # Plot the surface
    #ax.set_aspect('equal')
    ax.view_init(0, 0)
    ax.dist = 5.65
    if save:
        plt.savefig('spherePartion_' + str(numPartitions) + '.png')
