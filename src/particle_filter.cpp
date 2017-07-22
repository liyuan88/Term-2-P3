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
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    default_random_engine gen;
    
    num_particles=100; //number of particles
    
    cout << "particles=100";
    
    // Create the vector to contain the `num_particles` particles
    particles = vector<Particle>(num_particles);
    
    // Create the vector to contain the weight for each particle
    weights = vector<double>(num_particles);
    
    //Sensor measurement Noise
    normal_distribution<double> Noise_x(0, std[0]);
    normal_distribution<double> Noise_y(0, std[1]);
    normal_distribution<double> Noise_theta(0, std[2]);
    
    //Initialize particles
    for (int i=0; i<num_particles; i++){
        Particle p;
        p.id=i;
        p.x=x+Noise_x(gen);
        p.y=y+Noise_y(gen);
        p.theta=theta+Noise_theta(gen);
        p.weight=1.0;
        particles.push_back(p);
        
    }
    
    is_initialized=true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    //default_random_engine gen;
    //Noise
    default_random_engine gen;
    
    normal_distribution<double> Noise_x(0, std_pos[0]);
    normal_distribution<double> Noise_y(0, std_pos[1]);
    normal_distribution<double> Noise_theta(0, std_pos[2]);
    
    //Predict particle position
    for (int i = 0; i < num_particles; i++) {
        
        // yaw_rate=0
        if (fabs(yaw_rate) < 0.0001) {
            particles[i].x += delta_t * velocity * cos(particles[i].theta);
            particles[i].y += delta_t * velocity * sin(particles[i].theta);
        }
        // yaw_rate!=0
        else {
            particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
            particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
            particles[i].theta += yaw_rate * delta_t;
        }
        //Add particle noise
        particles[i].x +=Noise_x(gen);
        particles[i].y +=Noise_y(gen);
        particles[i].theta +=Noise_theta(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for (int i = 0; i < observations.size(); i++) {
        
        LandmarkObs o = observations[i];
        //initialize min_dist with a big number
        double min_distance = numeric_limits<double>::max();
        int map_id=-1;
        
        for (int j=0; j<predicted.size(); j++){
            
            LandmarkObs p = predicted[j];
            double distance=dist(o.x, o.y, p.x, p.y);
            
            if (distance < min_distance) {
                min_distance = distance;
                map_id = p.id;
            }
        }
        
        observations[i].id=map_id;
        
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    
    for (int i=0; i<num_particles; i++) {
        double p_x = particles[i].x;
        double p_y = particles[i].y;
        double p_theta = particles[i].theta;
        
        vector<LandmarkObs> landmark_inrange;
        
        for (int j = 0; j<map_landmarks.landmark_list.size(); j++) {
            
            
            
            float l_x=map_landmarks.landmark_list[j].x_f;
            float l_y=map_landmarks.landmark_list[j].y_f;
            int l_id=map_landmarks.landmark_list[j].id_i;
            
            if (dist(p_x, p_y, l_x, l_y)<sensor_range) {
                LandmarkObs l={l_id, l_x, l_y};
                landmark_inrange.push_back(l);
            }
            
        }
        
        vector<LandmarkObs> ob_mapcoordinates;
        
        for (int j=0; j<observations.size(); j++){
            double map_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
            double map_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
            LandmarkObs l = {observations[j].id, map_x, map_y};
            ob_mapcoordinates.push_back(l);
        }
        dataAssociation(landmark_inrange,ob_mapcoordinates);
        //reset the weight
        double new_weight =1.0;
        
        double denominator = 2 * M_PI * std_landmark[0] * std_landmark[1];
        double x_deno = 2 * pow(std_landmark[0], 2);
        double y_deno = 2 * pow(std_landmark[1], 2);
        for (int l = 0; l < ob_mapcoordinates.size(); l++)
        {
            LandmarkObs obs = ob_mapcoordinates[l];
            LandmarkObs mu = landmark_inrange[obs.id];
            double mu_x;
            double mu_y;
            
            for (unsigned int m = 0; m < landmark_inrange.size(); m++)
            {
                if (landmark_inrange[m].id == obs.id)
                {
                    mu_x = landmark_inrange[m].x;
                    mu_y = landmark_inrange[m].y;
                }
            }
            
            double x_diff = obs.x - mu_x;
            double y_diff = obs.y - mu_y;
            
            
            
            new_weight *= exp(-(pow(x_diff, 2) / x_deno) - (pow(y_diff, 2) / y_deno)) / denominator;
            
            
        }
        
        particles[i].weight = new_weight;
        weights[i] = particles[i].weight;
        
        /*for (int j=0; j<ob_mapcoordinates.size(); j++){
            double ob_x=ob_mapcoordinates[j].x - p_x;
            double ob_y=ob_mapcoordinates[j].y - p_y;
            double ob_length = sqrt(ob_x * ob_x + ob_y * ob_y);
            double ob_angle = atan2(ob_y, ob_x);
            int map_index = ob_mapcoordinates.id;
            double map_x=landmark_inrange[map_index].x - p_x;
            double map_y=landmark_inrange[map_index].y - p_y;
            double map_length = sqrt(map_x * map_x + map_y * map_y);
            double map_angle = atan2(map_y, map_x);
            double delta_length=ob_length-map_length;
            double delta_angle=ob_angle-map_angle;
            
            // Bivariate Gaussian
            double num_a = delta_length * delta_length / (2.0 * std_landmark[0] * std_landmark[0]);
            double num_b = delta_angle * delta_angle / (2.0 * std_landmark[1] * std_landmark[1]);
            double numerator = exp(-1.0 * (num_a + num_b));
            double denominator = 2.0 * M_PI * std_landmark[0] * std_landmark[1];
            new_weight *= numerator / denominator;
            
            cout<<"numerator ";
            cout<<numerator;
            cout<<"denominator ";
            cout<<denominator;
            
        }
        particles[i].weight = new_weight;
        weights[i] = particles[i].weight;
        //cout<<weights[i];
        */
    }
    
    
    
    
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine gen;
    discrete_distribution<>dist_particles(weights.begin(),weights.end());
    vector<Particle> new_particles(num_particles);
    for (int i=0; i<num_particles; i++){
        new_particles[i] = particles[dist_particles(gen)];
    }
    particles = new_particles;
    
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

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
