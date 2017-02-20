import numpy as np
import copy

# Main class to store trajectories and calculate discrete state trajectories for milestoning 
class allTrajs(object):
    
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
        self.entry_div = 30
        # Number of rings in the exit states
        self.rentry1 = 1.8
        self.rentry2 = 2.0
        # Calculated variables:
        # Angular increments
        self.angint_entry = 2*np.pi/self.entry_div
    
    # Get discretized trajectories (dTrajs) in chosen milestones 
    # from continue trajectories (Trajs)
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
        return self.dTrajs
    
    # Same as getdTrajs but ensuring there are no "None" states,
    # since they can appear if the initial condition is in a "None" state region
    def getdTrajsclean(self):
        # If dTrajs haven't been yet calculated, do so
        if self.dTrajs == []:
            self.getdTrajs()
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
    # uniformly theparticle in the exit states in the hybrid model
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


# Define child class to include entry and exit rings
class allTrajs_entry_exit_rings(allTrajs):
    
    def __init__(self, Trajs=None):
	super(allTrajs_entry_exit_rings, self).__init__(Trajs)
        # Milestone choice variables
        self.entry_div = 25
        self.exit_div = 25
        # Number of rings in the exit states
        self.rexit_div = 1
        self.rentry1 = 1.8
        self.rentry2 = 2.0
        self.rexit1 = 2.3
        self.rexit2 = 2.5
        # Calculated variables:
        # Angular increments
        self.angint_entry = 2*np.pi/self.entry_div
        self.angint_exit = 2*np.pi/self.exit_div
        # Radial increment
        self.rint_exit = (self.rexit2-self.rexit1)/float(self.rexit_div)
    
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
        # Exit states
        elif (r >= self.rexit1 and r <= self.rexit2):
            # Check in which ring the particle is
            for j in range(self.rexit_div):
                rincrement = (self.rexit2 - self.rexit1)/float(self.rexit_div)
                if (r >= self.rexit1 + j*self.rint_exit and r <= self.rexit1 + (j+1)*self.rint_exit):
                    for k in range(self.exit_div):
                        llim = -np.pi + k*self.angint_exit
                        rlim = -np.pi + (k+1)*self.angint_exit
                        if (th >= llim and th < rlim):
                            state = self.entry_div + j*self.exit_div + k + 1
                            return state
        # Bath state
        elif (r > self.rexit2):
            state = self.entry_div + self.rexit_div*self.exit_div + 1
            return state
        # Didn't change state
        else:
            state = prevst
            return state
    
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
        # Loope over exit states
        for j in range(self.rexit_div):
            for k in range(self.exit_div):
                llim = -np.pi + k*self.angint_exit
                rlim = -np.pi + (k+1)*self.angint_exit
                th = (rlim + llim)/2.0
                rexit = self.rexit1 + (j+0.5)*self.rint_exit
                x = rexit*np.cos(th)
                y = rexit*np.sin(th)
                self.milestones[k + 1 + self.entry_div + j*self.exit_div] = [x,y]
        self.milestones[self.entry_div + self.rexit_div*self.exit_div + 1] = [2.5,2.5] # Change if dim of system changes
        return self.milestones

    
    # Calculate State discretization parameters into a dictionary to place  
    # uniformly theparticle in the exit states in the hybrid model
    def getRegionMap(self):
        self.regionMap[0] = 'Bound'
        # Add entry states to dictionary
        for k in range(self.entry_div):
            llim = -np.pi + k*self.angint_entry
            rlim = -np.pi + (k+1)*self.angint_entry
            self.regionMap[k+1] = [[llim,rlim],[self.rentry1, self.rentry2]]
        # Add exit states to dictionary
        for j in range(self.rexit_div):
            for k in range(self.exit_div):
                llim = -np.pi + k*self.angint_exit
                rlim = -np.pi + (k+1)*self.angint_exit
                rllim = self.rexit1 + j*self.rint_exit
                rrlim = self.rexit1 + (j+1)*self.rint_exit
                self.regionMap[k + 1 + self.entry_div + j * self.exit_div] = [[llim, rlim],[rllim, rrlim]]
        # Add exit states radii interval
        self.regionMap['rexit_int'] = [self.rexit1, self.rexit2]
        self.regionMap['rentry_int'] = [self.rentry1, self.rentry2]
        return self.regionMap

