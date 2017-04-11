#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <stdio.h>
#include <vector>
#include <string>
# include <omp.h>
using namespace std;

// Include required functions file
#include "utils.h"
#include "particles.h"

int main()
{
  // Set seed of rand() as time dependent
  srand( time( NULL ) );
  
  // Initialization variables
  string runid = "0";
  string filename;
  int nsteps = 6000000;
  int numPart = 10000000;
  double PI = 1.0*M_PI;
  double diff = 1.0;
  double maxR = 4.0; //35
  double innerMSMRad = 2.0;
  double dt = 0.0001;
  double dr = 0.01; //sqrt(2*diff*dt);
  //string outerBoundary = "reflective";
  //int numKilledPart = 0;
  
  // Variables for spatial discretization
  int angularPartitions = 20;
  double dtheta = 2*PI/angularPartitions;
  
  
  // Print parameters used before starting simulation
  std::cout <<"Run simulation with dt:= " << dt << "\n";
  std::cout <<"Inner MSM radius:= " << innerMSMRad << "\n";
  std::cout <<"Max radius:= " << maxR << "\n"; // " with " << outerBoundary << "boundary conditions" << "\n";
  std::cout <<"Number of angular states:= " << angularPartitions << "\n";
  std::cout <<"Press any key to run simulation" << "\n";
  cin.get();
  
  
  // Temporay variables
  double posx;
  double posy;
  double prob;
  double inRad;
  
  // Create one particle 
  //Particle2D p1;

  // // Create vector of particles and temporary array for resizing
  //vector<Particle2D> particlearray;
  //particlearray.resize(numPart); 
  //vector<Particle2D> partarraytemp;
  
  // Arrays to save output data
  vector<int> exitTimes;
  vector<int> exitStates;
  exitTimes.resize(numPart);
  exitStates.resize(numPart); 
  //int exitTimes[ numPart ];
  //int exitStates[ numPart ];


//   // Assign initial conditions
//   for (int i = 0; i<numPart; i++) {
//   	particlearray[i].UniformRingSectionIC(innerMSMRad, innerMSMRad + dr, 0.0, dtheta);
//   }
  
  
  // Loop over all the time steps
  //int leftOverPart = numPart;
  int ii = 1;
  int partIsAlive;
  double rad;
  double state;
  double th;
  int numPartDone = 0;
  //omp not working slower than serial, need to check
//   omp_set_num_threads(4);
//   #pragma omp parallel for private(ii,partIsAlive,rad,state,th)
  for (int j = 0; j < numPart; j++){
      Particle2D part;
      part.UniformRingSectionIC(innerMSMRad, innerMSMRad + dr, 0.0, dtheta);
      ii = 1;
      partIsAlive = 1;
      while (partIsAlive == 1){
        part.MoveParticle(dt,diff);
        rad = part.GetRad();
        if (rad <=innerMSMRad) {
            // Find entry state
            th = part.GetTheta();
            for(int k = 0; k < angularPartitions; k++){
                if (th >= k*dtheta and th <= (k+1)*dtheta) {
                    state = k;
                    break;
                }
            }
            exitTimes[j] = ii;
            exitStates[j] = state;
            partIsAlive = 0;
            numPartDone++;
        }
        else if (rad > maxR) {
            exitTimes[j] = ii;
            exitStates[j] = -1;
            partIsAlive = 0;
            numPartDone++;
        }
        ii++;
      }
      // Print current state of the simulation
      if ((numPartDone)%100000 == 0) {
        std::cout <<"Trajectories finished: " << numPartDone << " of " << numPart << endl;
      }
  }
  
  
  
//   while (leftOverPart > 0){
//         
//         // Check for particles crashing into innerMSM or escaping
//         for(int j = 0; j < particlearray.size(); j++ ) {
//             particlearray[j].MoveParticle(dt,diff);
//             rad = particlearray[j].GetRad();
//             if (rad <=innerMSMRad) {
//                 // Find entry state
//                 th = particlearray[j].GetTheta();
//                 for(int k = 0; k < angularPartitions; k++){
//                     if (th >= k*dtheta and th <= (k+1)*dtheta) {
//                         state = k;
//                         break;
//                     }
//                 }
//                 exitTimes[numKilledPart] = ii;
//                 exitStates[numKilledPart] = state;
//                 numKilledPart++;
//                 particlearray[j].Kill();                
//             }
//             else if (rad > maxR) {
//                 exitTimes[numKilledPart] = ii;
//                 exitStates[numKilledPart] = -1;
//                 numKilledPart++;
//                 particlearray[j].Kill();
//             }
//         }
//         
//         // Update array by saving alive particles into partarraytemp
//         partarraytemp.resize(0);
//         for(int jj = 0; jj < particlearray.size(); jj++ ) {
//                 if (particlearray[jj].IsAlive() == 1) {
//                         partarraytemp.push_back(particlearray[jj]);
//                 }
//         }
//         
//         // Update original particle array
//         particlearray.resize(partarraytemp.size());
//         particlearray.swap(partarraytemp);
//         leftOverPart = particlearray.size();
//         
//         // Print current state of the simulation
//         if ((ii-1)%1000 == 0) {
//             std::cout <<"Step " << ii << " num of particles: " << leftOverPart << endl;
//         }
//     ii++;
//   }

  
  // Print average particle number to file
  filename = "data/parameters_" + runid + ".dat";
  ofstream paramdata (filename.c_str());
  paramdata << "[myparams]" << "\n";
  paramdata << "Simulation steps: " << nsteps << "\n";
  paramdata << "Diffusion coefficient: " << diff << "\n";
  paramdata << "Reaction radius: " << innerMSMRad << "\n";
  paramdata << "Maximum radius: " << maxR << "\n";
  paramdata << "Time interval dt: " << dt << "\n";
  paramdata << "Ring state width dr: " << dr << "\n";  
  paramdata.close();
  
  // Print exitTimes and exitState to file
  filename = "data/trajectories_" + runid + ".dat";
  ofstream pnumber (filename.c_str());
  for (int i=0; i < numPart; i++) {
    pnumber << exitTimes[i] << " " << exitStates[i] << "\n"; 
  }
  pnumber.close();
  
  std::cout <<"Simulation finished." << "\n";
  std::cout <<"Now calculating MFPTs" << "\n";
  
  double MFPTs[angularPartitions + 1];
  double rates[angularPartitions + 1];
  double timeSum;
  int totalSamples;
  // Calculate and print MFPTs to file
  filename = "data/MFPTs_" + runid + ".dat";
  ofstream allmfpts (filename.c_str());
  for (int i = 0; i < angularPartitions + 1; i++){
    // Check if in outstate or inner state
    if (i == angularPartitions){
        ii = -1;
    }
    else {
        ii = i;
    }
    timeSum = 0;
    totalSamples = 0;
    for (int j = 0; j < numPart; j++){
      if (exitStates[j] == ii) {
          timeSum = timeSum + exitTimes[j];
          totalSamples++;
      }
    }
    if (totalSamples == 0){
        MFPTs[i] = -1;
        rates[i] = 0;
    }
    else {
        MFPTs[i] = dt*timeSum/totalSamples;
        rates[i] = 1.0/MFPTs[i];
    }
    allmfpts << ii << " " << MFPTs[i] << " " << rates[i] << "\n"; 
  }
  allmfpts.close();
  std::cout <<"Done!" << "\n";
}
