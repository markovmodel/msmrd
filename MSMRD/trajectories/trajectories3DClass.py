import numpy as np
import copy

# Main class to store trajectories and calculate discrete state trajectories for milestoning 
class all3DTrajs(object):
    
    def __init__(self, Trajs=None):
        if Trajs == None:
            Trajs = []
        # Main variables
        self.Trajs = Trajs
        self.dTrajs = []
        self.dTrajsclean = []
        self.milestones = {}
        self.milestonesarray = []
        self.regionMap = {}
        # Milestone choice variables
        self.angular_divisions = 30
        # Number of rings in the exit states
        self.inner_MSMrad = 2.0
        self.rentry = 2.4
        self.rexit = 2.8
        # Calculated variables:
        # Angular increments
        self.angint_entry = 2*np.pi/self.entry_div
    
    # Get discretized trajectories (dTrajs) in chosen milestones 
    # from continue trajectories (Trajs) and clean None entries
    def getdTrajs(self):
        # Resize dTrajs array 
        self.dTrajs = [None] * len(self.Trajs)
        # Loop over each trajectory
        for i in range(len(self.Trajs)):
            # Create empty list of consistent size for ith discrete trajectory
            trajlen = len(self.Trajs[i])
            self.dTrajs[i] = [None] * trajlen
            # Loop over each time iteration to set corresponding discrete state
            for j in range(trajlen):
                if j > 0:
                    prevstate = self.dTrajs[i][j-1]
                else:
                    prevstate = None
                self.dTrajs[i][j] = self.getState(self.Trajs[i][j],prevstate)
            self.dTrajsclean = copy.deepcopy(self.dTrajs)
            # Eliminate "None" entries in reverse order to avoid misindexing
            for i in reversed(range(len(self.dTrajs))):
                if self.dTrajs[i] == None:
                    self.dTrajsclean.pop(i)
                else:
                    for j in reversed(range(len(self.dTrajs[i]))):
                        if self.dTrajs[i][j] == None:
                            self.dTrajsclean[i].pop(j)
                if self.dTrajsclean[i] == []:
                    self.dTrajsclean.pop(i)
        return self.dTrajsclean
    

    # Given coordinates, assigns a state which corresponds to an area
    # in space. The state is assigned with an integer value. The center of the
    # state region is given by getMilestones() function
    def getState(self, coord, prevst):
        x = coord[0]
        y = coord[1]
        r = np.sqrt(x*x + y*y)
        th = np.arctan2(y, x)
        # Bound state
        if r <= 1.:
            state = 0
            return state
        # Entry states
        elif (r >= self.rentry1 and r < self.rentry2):
            for k in range(self.entry_div):
                llim = -np.pi + k*self.angint_entry
                rlim = -np.pi + (k+1)*self.angint_entry
                if (th >= llim and th < rlim):
                    state = k + 1
                    return state
        # Bath state
        elif (r >= self.rentry2):
            state = self.entry_div + 1
            return state
        # Didn't change state
        else:
            state = prevst
            return state
        
        
    # Calculate equal area partition of sphere with num_partiions partitions
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
        ideal_num_regions = np.zeros(num_collars)
        phis = np.zeros(num_collars + 1)
        num_regions = np.zeros(num_collars)
        thetas = []
        a = [0]
        # Iterate over each collar to get right number of regions per collar
        # and correct location of phi angles of each collar
        for i in range(num_collars):
            # Calculate num of regions in collar i
            cap_area_phi1 = angle_to_cap_area(phi0 + i*collar_angle)
            cap_area_phi2 = angle_to_cap_area(phi0 + (i+1)*collar_angle)
            ideal_num_regions[i] = (cap_area_phi2 - cap_area_phi1)/state_area
            num_regions[i] = np.round(ideal_num_regions[i] + a[i])
            # Correct values of phi around collar i
            suma = 0
            for j in range(i+1):
                suma = suma + ideal_num_regions[j] - num_regions[j]
            a.append(suma)
            summ = 1
            for j in range(i):
                summ = summ + num_regions[j]
            phis[i] = cap_area_to_angle(summ*state_area)
            phis[-1] = np.pi - phi0
            # Obtain list of thetas for a given collar
            thetasi = []
            dth = 2.0*np.pi/num_regions[i]
            for j in range(int(num_regions[i])):
                thetasi.append(j*dth)
            thetas.append(thetasi)
            # return number of regions for all collars, 
            # phi angles of collars and theta angles for each collar
        return num_regions, phis, thetas 
    
    
    # Get x,y centers of milestones in a dictionary: milestones[state] = [x,y]    
    def getMilestones(self):
        # Bound state is 0 and assigned origin as center
        self.milestones[0] = [0.0,0.0]
        rentry = (self.rentry1 + self.rentry2)/2.0
        # Loop over entry states
        for k in range(self.entry_div):
            llim = -np.pi + k*self.angint_entry
            rlim = -np.pi + (k+1)*self.angint_entry
            th = (rlim + llim)/2.0
            x = rentry*np.cos(th)
            y = rentry*np.sin(th)
            self.milestones[k+1] = [x,y]
        self.milestones[self.entry_div + 1] = [2.5,2.5] # Change if dim of system changes
        return self.milestones
    
    # Get milestones centers in array for plotting
    def getMilestonesArray(self):
        if self.milestones == {}:
            self.getMilestones()
        self.milestonesarray = np.zeros((len(self.milestones),2))
        for i in range(len(self.milestones)):
            self.milestonesarray[i][0] = self.milestones[i][0]
            self.milestonesarray[i][1] = self.milestones[i][1]
        return self.milestonesarray
    
    # Calculate State discretization parameters into a dictionary to place  
    # uniformly the particle in the exit states in the hybrid model
    def getRegionMap(self):
        self.regionMap[0] = 'Bound'
        # Add entry states to dictionary
        for k in range(self.entry_div):
            llim = -np.pi + k*self.angint_entry
            rlim = -np.pi + (k+1)*self.angint_entry
            self.regionMap[k+1] = [[llim,rlim],[self.rentry1, self.rentry2]]
        # Add entry states radii interval
        self.regionMap['rentry_int'] = [self.rentry1, self.rentry2]
        return self.regionMap

    # Extract trajectories from truncated radius
    def extractTraj(self, truncRad = 2.0):
        self.extractedTrajs = []
        for traj in self.Trajs:
            norm = np.linalg.norm(traj, axis = 1)
            trajActive = False
            for i in range(len(traj)):
                if not trajActive:
                    if norm[i] < truncRad:
                        #Start a new inner truncated trajectory
                        trajActive = True
                        currentInnerTraj = [traj[i]]
                else:
                    if norm[i] > truncRad:
                        trajActive = False
                        currentInnerTraj.append(traj[i])
                        self.extractedTrajs.append(np.array(currentInnerTraj))
                    else:
                        currentInnerTraj.append(traj[i])
        return self.extractedTrajs


