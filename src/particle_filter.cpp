/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Set the number of particles
  num_particles = 100;

  // Creates a normal (Gaussian) distribution for x.
  normal_distribution<double> dist_x(x, std[0]);
  // Creates a normal (Gaussian) distribution for y.
  normal_distribution<double> dist_y(y, std[1]);
  // Creates a normal (Gaussian) distribution for theta.
  normal_distribution<double> dist_theta(theta, std[2]);

  // Initialize all particles to first position (based on estimates of 
  //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
  default_random_engine gen;
  for (int i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.;
    particles.push_back(p);
    weights.push_back(p.weight);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;
  for (int i = 0; i < num_particles; i++) {
    Particle p = particles[i];
    double x = p.x;
    double y = p.y;
    double theta = p.theta;
    // Add measurements to the particle
    if (fabs(yaw_rate) < 0.0001) {
      x += velocity * cos(p.theta) * delta_t;
      y += velocity * sin(p.theta) * delta_t;
    }
    else {
      x += velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
      y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
      theta += yaw_rate * delta_t;
    }

    /* add random Gaussian noise */
    // Creates a normal (Gaussian) distribution for x.
    normal_distribution<double> dist_x(x, std_pos[0]);
    // Creates a normal (Gaussian) distribution for y.
    normal_distribution<double> dist_y(y, std_pos[1]);
    // Creates a normal (Gaussian) distribution for theta.
    normal_distribution<double> dist_theta(theta, std_pos[2]);

    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);

    particles[i] = p;
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	// observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  for (int i = 0; i < observations.size(); i ++) {
    LandmarkObs *obs = &observations[i];
    LandmarkObs *pred = &predicted[0];
    double closest_dist = dist(obs->x, obs->y, pred->x, pred->y);
    obs->id = pred->id;
    for (int j = 1; j < predicted.size(); j++) {
      pred = &predicted[j];
      double the_dist = dist(obs->x, obs->y, pred->x, pred->y);
      if (closest_dist > the_dist) {
        closest_dist = the_dist;
        obs->id = pred->id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution

	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  weights.clear();
  for (int i = 0; i < num_particles; i++) {
    Particle *p = &particles[i];
    vector<LandmarkObs> predicted;
    vector<LandmarkObs> map_observations;
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;

    // Find the predicated landmarks for this partical within the sensor range
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      LandmarkObs lm;
      lm.id = map_landmarks.landmark_list[j].id_i;
      lm.x = map_landmarks.landmark_list[j].x_f;
      lm.y = map_landmarks.landmark_list[j].y_f;
      double d = dist(p->x, p->y, lm.x, lm.y);
      if (d <= sensor_range) {
        predicted.push_back(lm);
        associations.push_back(lm.id);
        sense_x.push_back(lm.x);
        sense_y.push_back(lm.y);
      }
    }
    SetAssociations(particles[i], associations, sense_x, sense_y);

    // Convert the observations from the VEHICLE's coordinate system to the MAP's coordinate system
    for (int j = 0; j < observations.size(); j++) {
      LandmarkObs Z = observations[j];
      double ob_x = Z.x;
      double ob_y = Z.y;
      Z.x = ob_x * cos(p->theta) - ob_y * sin(p->theta) + p->x;
      Z.y = ob_y * cos(p->theta) + ob_x * sin(p->theta) + p->y;
      map_observations.push_back(Z);
    }

    dataAssociation(predicted, map_observations);

    double w = 1;
    for (int j = 0; j < map_observations.size(); j++) {
      double std_x = std_landmark[0];
      double std_y = std_landmark[1];
      double ob_x = map_observations[j].x;
      double ob_y = map_observations[j].y;
      int map_id = map_observations[j].id;
      double pred_x = map_landmarks.landmark_list[map_id - 1].x_f;
      double pred_y = map_landmarks.landmark_list[map_id - 1].y_f;
      // calculate exponent
      double exponent = -1. / (2. * std_x * std_x) * (pred_x - ob_x) * (pred_x - ob_x) -1. / (2. * std_y * std_y) * (pred_y - ob_y) * (pred_y - ob_y);
      w *= exp(exponent)/ (2. * M_PI * std_x * std_y);
    }

    p->weight = w;
    weights.push_back(w);
  }
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  
  default_random_engine gen;

  std::discrete_distribution<> dist_w(weights.begin(), weights.end());
  std::vector<Particle> resampled_particles;

  for (int i = 0; i < num_particles; i ++) {
    Particle p = particles[dist_w(gen)];
    p.id = i;
    resampled_particles.push_back(p);
  }

  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
