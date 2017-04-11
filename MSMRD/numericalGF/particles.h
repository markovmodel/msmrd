class Particle2D {
private:
   double posx;      // Data member (Variable)
   double posy;
   int alive;

public:
   // Constructor with default values for data members
   Particle2D(double px = 10, double py = 10, int ali = 1) {
      posx = px;
      posy = py;
      alive = ali;
   }
 
   void MoveParticle(double dt, double diff) {    // Member function
    double wpcoeff = sqrt(2*dt*diff);
    double rnorm1 = rand_normal(0,1);
    double rnorm2 = rand_normal(0,1);       
    posx = posx + wpcoeff*rnorm1;
        posy = posy + wpcoeff*rnorm2;   
   }

   void UniformDiskIC(double maxRad) {
    double randr = 1.0*rand()/RAND_MAX;
        double randth = 2.0*M_PI*rand()/RAND_MAX;
        posx = maxRad*sqrt(randr)*cos(randth);
        posy = maxRad*sqrt(randr)*sin(randth);
   }

   void UniformRingIC(double inR, double outR) {
    double randr = 1.0*rand()/RAND_MAX;
        double randth = 2.0*M_PI*rand()/RAND_MAX;
        double zz = randr*(outR*outR - inR*inR) + inR*inR;
        posx = sqrt(zz)*cos(randth);
        posy = sqrt(zz)*sin(randth);
   }
   
   void UniformRingSectionIC(double inR, double outR, double theta1, double dtheta) {
        double randr = 1.0*rand()/RAND_MAX;
        double randth = theta1 + dtheta*rand()/RAND_MAX;
        double zz = randr*(outR*outR - inR*inR) + inR*inR;
        posx = sqrt(zz)*cos(randth);
        posy = sqrt(zz)*sin(randth);
   }

   double GetPosX() {
    return posx;
   }

   double GetPosY() {
    return posy;
   }

   double GetRad() {
    return sqrt(posx*posx + posy*posy);
   }
   
   double GetTheta() {
    // Return angle between 0,2pi
    double theta = atan2(posy, posx);
    if (theta < 0) {
        theta = theta + 2*M_PI;
    }
    return theta;
   }

   int IsAlive(){
    return alive;
   }

   void Kill() {
    alive = 0;
   }
   
   void Revive() {
    alive = 1;
   }  
};   // need to end the class declaration with a semi-colon


class Particle3D {
private:
   double posx;      // Data member (Variable)
   double posy;
   double posz;
   double prevx;
   double prevy;
   double prevz;
   int alive;

public:
   // Constructor with default values for data members
      Particle3D(double px = 10, double py = 10, double pz = 10, int ali = 1) {
      posx = px;
      posy = py;
      posz = pz;
      alive = ali;
   }
 
   void MoveParticle(double dt, double diff) {    // Member function
        double wpcoeff = sqrt(2*dt*diff);
        double rnorm1 = rand_normal(0,1);
        double rnorm2 = rand_normal(0,1);
        double rnorm3 = rand_normal(0,1);
        prevx = posx;
        prevy = posy;
        prevz = posz;
        posx = posx + wpcoeff*rnorm1;
        posy = posy + wpcoeff*rnorm2;  
        posz = posz + wpcoeff*rnorm3; 
   }

   void UniformSphereIC(double maxRad) {
        double randr = 1.0*rand()/RAND_MAX;
        double randcosth = 2.0*rand()/RAND_MAX - 1.0;
        double randph = 2.0*M_PI*rand()/RAND_MAX;
        double rr = maxRad*cbrt(randr);
        double th = acos(randcosth);
        posx = rr*sin(th)*cos(randph);
        posy = rr*sin(th)*sin(randph);
        posz = rr*cos(th);
   }

   void UniformSphericalShellIC(double inR, double outR) {
       // Sample in sphere and do rejection sampling
       bool outshell = true;
       double rr;
       while (outshell) {
        rr = 1.0*rand()/RAND_MAX;
        rr = outR*cbrt(rr);
        if (rr >= inR && rr < outR){
            outshell = false;
            }    
        }
        double randcosth = 2.0*rand()/RAND_MAX - 1.0;
        double randph = 2.0*M_PI*rand()/RAND_MAX;
        double th = acos(randcosth);
        posx = rr*sin(th)*cos(randph);
        posy = rr*sin(th)*sin(randph);
        posz = rr*cos(th);
        // NOT SURE IF OPTION B IS 100% CORRECT
        /*double randr = 1.0*rand()/RAND_MAX;
        double randcosth = 2.0*rand()/RAND_MAX - 1.0;
        double randph = 2.0*M_PI*rand()/RAND_MAX;
        double rr = cbrt(randr*(pow(outR,3)-pow(inR,3))+pow(inR,3)); //not 100% sure
        double th = acos(randcosth);
        posx = rr*sin(th)*cos(randph);
        posy = rr*sin(th)*sin(randph);
        posz = rr*cos(th); */
   }
   
   void Reflect() {
    // NEED TO WRITE FULL FUNCTION TO REFLECT PARTICLE
       posx = prevx;
       posy = prevy;
       posz = prevz;
    }

   double GetPosX() {
        return posx;
   }

   double GetPosY() {
        return posy;
   }
   
   double GetPosZ() {
        return posz;
   }

   double GetRad() {
        return sqrt(posx*posx + posy*posy + posz*posz);
   }

   int IsAlive(){
        return alive;
   }

   void Kill() {
        alive = 0;
   }  
};