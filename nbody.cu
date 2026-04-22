#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

//double G = 1;

struct simulation {
  size_t nbpart;
  
  std::vector<double> mass;

  //position
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;

  //velocity
  std::vector<double> vx;
  std::vector<double> vy;
  std::vector<double> vz;

  //force
  std::vector<double> fx;
  std::vector<double> fy;
  std::vector<double> fz;

  
  simulation(size_t nb)
    :nbpart(nb), mass(nb),
     x(nb), y(nb), z(nb),
     vx(nb), vy(nb), vz(nb),
     fx(nb), fy(nb), fz(nb) 
  {}
};


void random_init(simulation& s) {
  std::random_device rd;  
  std::mt19937 gen(rd());
  std::uniform_real_distribution dismass(0.9, 1.);
  std::normal_distribution dispos(0., 1.);
  std::normal_distribution disvel(0., 1.);

  for (size_t i = 0; i<s.nbpart; ++i) {
    s.mass[i] = dismass(gen);

    s.x[i] = dispos(gen);
    s.y[i] = dispos(gen);
    s.z[i] = dispos(gen);
    s.z[i] = 0.;
    
    s.vx[i] = disvel(gen);
    s.vy[i] = disvel(gen);
    s.vz[i] = disvel(gen);
    s.vz[i] = 0.;
    s.vx[i] = s.y[i]*1.5;
    s.vy[i] = -s.x[i]*1.5;
  }

  return;
  //normalize velocity (using normalization found on some physicis blog)
  double meanmass = 0;
  double meanmassvx = 0;
  double meanmassvy = 0;
  double meanmassvz = 0;
  for (size_t i = 0; i<s.nbpart; ++i) {
    meanmass += s.mass[i];
    meanmassvx += s.mass[i] * s.vx[i];
    meanmassvy += s.mass[i] * s.vy[i];
    meanmassvz += s.mass[i] * s.vz[i];
  }
  for (size_t i = 0; i<s.nbpart; ++i) {
    s.vx[i] -= meanmassvx/meanmass;
    s.vy[i] -= meanmassvy/meanmass;
    s.vz[i] -= meanmassvz/meanmass;
  }
  
}

void init_solar(simulation& s) {
  enum Planets {SUN, MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE, MOON};
  s = simulation(10);

  // Masses in kg
  s.mass[SUN] = 1.9891 * std::pow(10, 30);
  s.mass[MERCURY] = 3.285 * std::pow(10, 23);
  s.mass[VENUS] = 4.867 * std::pow(10, 24);
  s.mass[EARTH] = 5.972 * std::pow(10, 24);
  s.mass[MARS] = 6.39 * std::pow(10, 23);
  s.mass[JUPITER] = 1.898 * std::pow(10, 27);
  s.mass[SATURN] = 5.683 * std::pow(10, 26);
  s.mass[URANUS] = 8.681 * std::pow(10, 25);
  s.mass[NEPTUNE] = 1.024 * std::pow(10, 26);
  s.mass[MOON] = 7.342 * std::pow(10, 22);

  // Positions (in meters) and velocities (in m/s)
  double AU = 1.496 * std::pow(10, 11); // Astronomical Unit

  s.x = {0, 0.39*AU, 0.72*AU, 1.0*AU, 1.52*AU, 5.20*AU, 9.58*AU, 19.22*AU, 30.05*AU, 1.0*AU + 3.844*std::pow(10, 8)};
  s.y = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  s.z = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  s.vx = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  s.vy = {0, 47870, 35020, 29780, 24130, 13070, 9680, 6800, 5430, 29780 + 1022};
  s.vz = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
}


void reset_force(simulation& s) {
  for (size_t i=0; i<s.nbpart; ++i) {
    s.fx[i] = 0.;
    s.fy[i] = 0.;
    s.fz[i] = 0.;
  }
}


void dump_state(simulation& s) {
  std::cout<<s.nbpart<<'\t';
  for (size_t i=0; i<s.nbpart; ++i) {
    std::cout<<s.mass[i]<<'\t';
    std::cout<<s.x[i]<<'\t'<<s.y[i]<<'\t'<<s.z[i]<<'\t';
    std::cout<<s.vx[i]<<'\t'<<s.vy[i]<<'\t'<<s.vz[i]<<'\t';
    std::cout<<s.fx[i]<<'\t'<<s.fy[i]<<'\t'<<s.fz[i]<<'\t';
  }
  std::cout<<'\n';
}

void load_from_file(simulation& s, std::string filename) {
  std::ifstream in (filename);
  size_t nbpart;
  in>>nbpart;
  s = simulation(nbpart);
  for (size_t i=0; i<s.nbpart; ++i) {
    in>>s.mass[i];
    in >>  s.x[i] >>  s.y[i] >>  s.z[i];
    in >> s.vx[i] >> s.vy[i] >> s.vz[i];
    in >> s.fx[i] >> s.fy[i] >> s.fz[i];
  }
  if (!in.good())
    throw "kaboom";
}

__global__ void compute_force_kernel(int n, const double* mass,const double *x, 
		
		const double* y, const double* z, 
		
		double *fx, double *fy, double *fz ) {
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return; 

	double localfx = 0.0;
	double localfy = 0.0;
	double localfz = 0.0;

	const double G = 6.674e-11;
	const double softening = 0.1;

	for (int j = 0; j < n; ++j){
		if (i == j) continue;
		
		double dx = x[j] - x[i];
		double dy = y[j] - y[i];
		double dz = z[j] - z[i];

    double dist_sq = dx * dx + dy * dy + dz * dz;
		double softened = dist_sq + softening;
    double F = G * mass[i] * mass[j] / softened;
    double norm = sqrt(softened);
		dx = dx / norm;
		dy = dy / norm;
		dz = dz / norm;

		localfx += dx * F;
		localfy += dy * F;
		localfz += dz * F;
	}
	fx[i] = localfx;
	fy[i] = localfy; 
	fz[i] = localfz;


}

__global__ void update_kernel(
	int n, const double* mass, double *x, double *y, double *z,
	double *vx, double *vy, double *vz, 
	const double *fx, const double *fy, const double *fz,
	double dt
	){

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= n) return;

	vx[i] += fx[i] / mass[i] * dt;
	vy[i] += fy[i] / mass[i] * dt;
	vz[i] += fz[i] / mass[i] * dt;

	x[i] += vx[i] * dt;
	y[i] += vy[i] * dt;
	z[i] += vz[i] * dt;

}

int main(int argc, char* argv[]) {
  if (argc != 6) {
    std::cerr
      <<"usage: "<<argv[0]<<" <input> <dt> <nbstep> <printevery>"<<"\n"
      <<"input can be:"<<"\n"
      <<"a number (random initialization)"<<"\n"
      <<"planet (initialize with solar system)"<<"\n"
      <<"a filename (load from file in singleline tsv)"<<"\n"
      <<"the number of threads you want to run"<<"\n";
    return -1;
  }
  
  double dt = std::atof(argv[2]); //in seconds
  size_t nbstep = std::atol(argv[3]);
  size_t printevery = std::atol(argv[4]);
  int blockSize = std::stoi(argv[5]);
  
  
  
  simulation s(1);

  //parse command line
  {
    size_t nbpart = std::atol(argv[1]); //return 0 if not a number
    if ( nbpart > 0) {
      s = simulation(nbpart);
      random_init(s);
    } else {
      std::string inputparam = argv[1];
      if (inputparam == "planet") {
	      init_solar(s);
      } else{
	      load_from_file(s, inputparam);
      }
    }    
  }
  int blocks = (s.nbpart + blockSize - 1) / blockSize;
  double *d_mass, *d_x, *d_y, *d_z;
  double *d_vx, *d_vy, *d_vz;
  double *d_fx, *d_fy, *d_fz;

  int n = s.nbpart;

  cudaMalloc(&d_mass, n * sizeof(double));
  cudaMalloc(&d_x,    n * sizeof(double));
  cudaMalloc(&d_y,    n * sizeof(double));
  cudaMalloc(&d_z,    n * sizeof(double));
  cudaMalloc(&d_vx,   n * sizeof(double));
  cudaMalloc(&d_vy,   n * sizeof(double));
  cudaMalloc(&d_vz,   n * sizeof(double));
  cudaMalloc(&d_fx,   n * sizeof(double));
  cudaMalloc(&d_fy,   n * sizeof(double));
  cudaMalloc(&d_fz,   n * sizeof(double));

  cudaMemcpy(d_mass, s.mass.data(), n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x,    s.x.data(),    n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y,    s.y.data(),    n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_z,    s.z.data(),    n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vx,   s.vx.data(),   n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vy,   s.vy.data(),   n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vz,   s.vz.data(),   n * sizeof(double), cudaMemcpyHostToDevice);
  dump_state(s);
  for (size_t step = 0; step < nbstep; step++) {
      if (step > 0 && step % printevery == 0) {
          cudaMemcpy(s.x.data(),  d_x,  s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
          cudaMemcpy(s.y.data(),  d_y,  s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
          cudaMemcpy(s.z.data(),  d_z,  s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
          cudaMemcpy(s.vx.data(), d_vx, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
          cudaMemcpy(s.vy.data(), d_vy, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
          cudaMemcpy(s.vz.data(), d_vz, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
          cudaMemcpy(s.fx.data(), d_fx, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
          cudaMemcpy(s.fy.data(), d_fy, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
          cudaMemcpy(s.fz.data(), d_fz, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);

          dump_state(s);
      }

      compute_force_kernel<<<blocks, blockSize>>>(
          s.nbpart, d_mass, d_x, d_y, d_z, d_fx, d_fy, d_fz
      );
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess)
          std::cerr << cudaGetErrorString(err) << '\n';

      update_kernel<<<blocks, blockSize>>>(
          s.nbpart, d_mass,
          d_x, d_y, d_z,
          d_vx, d_vy, d_vz,
          d_fx, d_fy, d_fz,
          dt
      );

      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess)
          std::cerr << cudaGetErrorString(err) << '\n';
      cudaDeviceSynchronize();
  }
  
  //dump_state(s);  

  cudaFree(d_mass);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaFree(d_vx);
  cudaFree(d_vy);
  cudaFree(d_vz);
  cudaFree(d_fx);
  cudaFree(d_fy);
  cudaFree(d_fz);
  return 0;
}
