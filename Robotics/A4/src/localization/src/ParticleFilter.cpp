#include "localization/ParticleFilter.h"
#include "localization/Util.h"

#include "tf/tf.h"

using namespace std;

ParticleFilter::ParticleFilter(int numberOfParticles) {
	this->numberOfParticles = numberOfParticles;

	// initialize particles
	for (int i = 0; i < numberOfParticles; i++) {
		this->particleSet.push_back(new Particle());
	}

	// this variable holds the estimated robot pose
	this->bestHypothesis = new Particle();

	// at each correction step of the filter only the laserSkip-th beam of a scan should be integrated
	this->laserSkip = 5;

	// distance map used for computing the likelihood field
	this->distMap = NULL;
}

ParticleFilter::~ParticleFilter() {
	// delete particles
	for (int i = 0; i < numberOfParticles; i++) {
		Particle* p = this->particleSet[i];
		delete p;
	}

	this->particleSet.clear();

	if (this->likelihoodField)
		delete[] this->likelihoodField;

	delete this->bestHypothesis;

	if (this->distMap)
		delete[] this->distMap;
}

int ParticleFilter::getNumberOfParticles() {
	return this->numberOfParticles;
}

std::vector<Particle*>* ParticleFilter::getParticleSet() {
	return &(this->particleSet);
}

void ParticleFilter::initParticlesUniform() {
    //get map properties
    int mapWidth, mapHeight;
    double mapResolution;
    this->getLikelihoodField(mapWidth, mapHeight,mapResolution);

	// TODO: here comes your code
	int totalInitParticles = this->getNumberOfParticles();

	double weight = 1.0 / (double)this->numberOfParticles;

	for (int i = 0; i < this->numberOfParticles; i++) {
		Particle* p = particleSet[i];
		
		p->x = Util::uniformRandom(0, mapWidth* mapResolution);
		p->y = Util::uniformRandom(0, mapHeight* mapResolution); 
		p->theta = Util::uniformRandom(0, 2*M_PI);
		p->weight = weight; 
	}
}

void ParticleFilter::initParticlesGaussian(double mean_x, double mean_y,
		double mean_theta, double std_xx, double std_yy, double std_tt) {
	// TODO: here comes your code
	double weight = 1.0 / (double)this->numberOfParticles;

	for (int i = 0; i < this->numberOfParticles; i++) {
		Particle* p = particleSet[i];
		p->x = Util::gaussianRandom(mean_x, std_xx);
		p->y = Util::gaussianRandom(mean_y, std_yy);
		p->theta = Util::normalizeTheta(Util::gaussianRandom(mean_theta, std_tt));
		p->weight = weight; 
	}
}

/**
 *  Initializes the likelihood field as our sensor model.
 */
void ParticleFilter::setMeasurementModelLikelihoodField(
		const nav_msgs::OccupancyGrid& map, double zRand, double sigmaHit) {
	ROS_INFO("Creating likelihood field for laser range finder...");

	// create the likelihood field - with the same discretization as the occupancy grid map
	this->likelihoodField = new double[map.info.height * map.info.width];
	this->likelihoodFieldWidth = map.info.width;
	this->likelihoodFieldHeight = map.info.height;
	this->likelihoodFieldResolution = map.info.resolution;

    // calculates the distance map and stores it in member variable 'distMap'
	// for every map position it contains the distance to the nearest occupied cell.
	calculateDistanceMap(map);

    // Here you have to create your likelihood field
	// HINT0: sigmaHit is given in meters. You have to take into account the resolution of the likelihood field to apply it.
	// HINT1: You will need the distance map computed 3 lines above
	// HINT2: You can visualize it in the map_view when clicking on "show likelihood field" and "publish all".
	// HINT3: Storing probabilities in each cell between 0.0 and 1.0 might lead to round-off errors, therefore it is
	// good practice to convert the probabilities into log-space, i.e. storing log(p(x,y)) in each cell. As a further
	// advantage you can simply add the log-values in your sensor model, when you weigh each particle according the
	// scan, instead of multiplying the probabilities, because: log(a*b) = log(a)+log(b).

	// TODO: here comes your code
	double zHit = 1.0 - zRand;

	double sigmaHitCells = sigmaHit / this->likelihoodFieldResolution;

	for (int y = 0; y < this->likelihoodFieldHeight; y++) {
		for (int x = 0; x < this->likelihoodFieldWidth; x++) {
			
			int idx = computeMapIndex(this->likelihoodFieldWidth, this->likelihoodFieldHeight, x, y);
			double dist = this->distMap[idx];

			// probability of a hit
			double pHit = Util::gaussian(dist, sigmaHitCells, 0.0);
			
			// total probability with random measurement component
			const double pRand = 1.0;
			double pMeasurement = zRand * pRand + zHit * pHit;

			// store log-probability to avoid round-off errors
			this->likelihoodField[idx] = log(pMeasurement);
		}
	}

	ROS_INFO("...DONE creating likelihood field!");
}

void ParticleFilter::calculateDistanceMap(const nav_msgs::OccupancyGrid& map) {
	// calculate distance map = distance to nearest occupied cell
	distMap = new double[likelihoodFieldWidth * likelihoodFieldHeight];
	int occupiedCellProbability = 90;
	// initialize with max distances
	for (int x = 0; x < likelihoodFieldWidth; x++) {
		for (int y = 0; y < likelihoodFieldHeight; y++) {
			distMap[x + y * likelihoodFieldWidth] = 32000.0;
		}
	}
	// set occupied cells next to unoccupied space to zero
	for (int x = 0; x < map.info.width; x++) {
		for (int y = 0; y < map.info.height; y++) {
			if (map.data[x + y * map.info.width] >= occupiedCellProbability) {
				bool border = false;
				for (int i = -1; i <= 1; i++) {
					for (int j = -1; j <= 1; j++) {
						if (!border && x + i >= 0 && y + j >= 0 && x + i
								< likelihoodFieldWidth && y + j
								< likelihoodFieldHeight && (i != 0 || j != 0)) {
							if (map.data[x + i + (y + j) * likelihoodFieldWidth]
									< occupiedCellProbability && map.data[x + i
									+ (y + j) * likelihoodFieldWidth] >= 0)
								border = true;
						}
						if (border)
							distMap[x + i + (y + j) * likelihoodFieldWidth]
									= 0.0;
					}
				}
			}
		}
	}
	// first pass -> SOUTHEAST
	for (int x = 0; x < likelihoodFieldWidth; x++)
		for (int y = 0; y < likelihoodFieldHeight; y++)
			for (int i = -1; i <= 1; i++)
				for (int j = -1; j <= 1; j++)
					if (x + i >= 0 && y + j >= 0 && x + i
							< likelihoodFieldWidth && y + j
							< likelihoodFieldHeight && (i != 0 || j != 0)) {
						double v = distMap[x + i + (y + j)
								* likelihoodFieldWidth] + ((i * j != 0) ? 1.414
								: 1);
						if (v < distMap[x + y * likelihoodFieldWidth]) {
							distMap[x + y * likelihoodFieldWidth] = v;
						}
					}

	// second pass -> NORTHWEST
	for (int x = likelihoodFieldWidth - 1; x >= 0; x--)
		for (int y = likelihoodFieldHeight - 1; y >= 0; y--)
			for (int i = -1; i <= 1; i++)
				for (int j = -1; j <= 1; j++)
					if (x + i >= 0 && y + j >= 0 && x + i
							< likelihoodFieldWidth && y + j
							< likelihoodFieldHeight && (i != 0 || j != 0)) {
						double v = distMap[x + i + (y + j)
								* likelihoodFieldWidth] + ((i * j != 0) ? 1.414
								: 1);
						if (v < distMap[x + y * likelihoodFieldWidth]) {
							distMap[x + y * likelihoodFieldWidth] = v;
						}
					}
}

double* ParticleFilter::getLikelihoodField(int& width, int& height,
		double& resolution) {
	width = this->likelihoodFieldWidth;
	height = this->likelihoodFieldHeight;
	resolution = this->likelihoodFieldResolution;

	return this->likelihoodField;
}

/**
 *  A generic measurement integration method that invokes some specific observation model.
 *  Maybe in the future, we add some other model here.
 */
void ParticleFilter::measurementModel(
		const sensor_msgs::LaserScanConstPtr& laserScan) {
	likelihoodFieldRangeFinderModel(laserScan);
}

/**
 *  Method that implements the endpoint model for range finders.
 *  It uses a precomputed likelihood field to weigh the particles according to the scan and the map.
 */
void ParticleFilter::likelihoodFieldRangeFinderModel(
		const sensor_msgs::LaserScanConstPtr & laserScan) {

	// TODO: here comes your code
	const int width = this->likelihoodFieldWidth;
    const int height = this->likelihoodFieldHeight;
    const double res = this->likelihoodFieldResolution;

	for (int i = 0; i < this->numberOfParticles; i++) {

		Particle* particle = this->particleSet[i];
		double logWeight = 0.0;

		for(int j = 0; j < laserScan->ranges.size(); j += this->laserSkip){
			double range = laserScan->ranges[j];

			if(isnan(range) || range > laserScan->range_max || range < laserScan->range_min){
				continue;
			}

			const double angle = laserScan->angle_min + j * laserScan->angle_increment;
			const double beamAngle = Util::normalizeTheta(particle->theta + angle);

			const double endX = particle->x + range * cos(beamAngle);
			const double endY = particle->y + range * sin(beamAngle);

			int mx = (int)(endX / this->likelihoodFieldResolution);
			int my = (int)(endY / this->likelihoodFieldResolution);

			if (mx >= 0 && my >= 0 && mx < width && my < height){
				int idx = computeMapIndex(this->likelihoodFieldWidth, this->likelihoodFieldHeight, mx, my);
                logWeight += this->likelihoodField[idx];
			} else{
				logWeight += log(0.0001); // penalize particles that project outside the map
			}
		}
		particle->weight = logWeight;
	}
}

void ParticleFilter::setMotionModelOdometry(double alpha1, double alpha2,
		double alpha3, double alpha4) {
	this->odomAlpha1 = alpha1;
	this->odomAlpha2 = alpha2;
	this->odomAlpha3 = alpha3;
	this->odomAlpha4 = alpha4;

}

/**
 *  A generic motion integration method that invokes some specific motion model.
 *  Maybe in the future, we add some other model here.
 */
void ParticleFilter::sampleMotionModel(double oldX, double oldY,
		double oldTheta, double newX, double newY, double newTheta) {
	sampleMotionModelOdometry(oldX, oldY, oldTheta, newX, newY, newTheta);
}

/**
 *  Method that implements the odometry-based motion model.
 */
void ParticleFilter::sampleMotionModelOdometry(double oldX, double oldY,
		double oldTheta, double newX, double newY, double newTheta) {
	// TODO: here comes your code
	// compute odometry based on Probabilistic Robotics Table 5.6
	double deltaX = newX - oldX;
	double deltaY = newY - oldY;
	
	double phi = atan2(deltaY, deltaX);
	double deltaRot1 = Util::normalizeTheta(Util::diffAngle(oldTheta, phi));
	double deltaTrans = sqrt(deltaX * deltaX + deltaY * deltaY);
	double deltaRot2 = Util::normalizeTheta(Util::diffAngle(oldTheta + deltaRot1, newTheta));
	
	// apply to each particle
	for (Particle* p : particleSet)	{
		const double rot1Hat = deltaRot1 + Util::gaussianRandom(0.0, this->odomAlpha1 * fabs(deltaRot1) + this->odomAlpha2 * fabs(deltaTrans));
		const double transHat = deltaTrans + Util::gaussianRandom(0.0, this->odomAlpha3 * deltaTrans + this->odomAlpha4 * (fabs(deltaRot1) + fabs(deltaRot2)));
		const double rot2Hat = deltaRot2 + Util::gaussianRandom(0.0, this->odomAlpha1 * fabs(deltaRot2) + this->odomAlpha2 * fabs(deltaTrans));

		p->x += transHat * cos(p->theta + rot1Hat);
		p->y += transHat * sin(p->theta + rot1Hat);
		p->theta = Util::normalizeTheta(p->theta + rot1Hat + rot2Hat);
	}
}

/**
 *  The stochastic importance resampling.
 */
void ParticleFilter::resample() {
	// TODO: here comes your code
	// compute total weight
	double sumW = 0.0;
    for (Particle* p : this->particleSet) {
        sumW += exp(p->weight);
    }

	if (sumW <= 0.0)
        return;

	// normalize weights
	Particle* bestParticle = nullptr;
	double maxWeight = -1.0;

    for (Particle* p : this->particleSet)
    {
        p->weight = exp(p->weight)/sumW;

        if (p->weight > maxWeight)
        {
            maxWeight = p->weight;
            bestParticle = p;
        }
    }

    // update bestHypothesis
    if (bestParticle)
    {
        *this->bestHypothesis = bestParticle;
		// this->bestHypothesis->x = bestParticle->x;
        // this->bestHypothesis->y = bestParticle->y;
        // this->bestHypothesis->theta = bestParticle->theta;
        // this->bestHypothesis->weight = bestParticle->weight;
    }

	// generate cdf
	vector<double> c(this->getNumberOfParticles(), 0.0);
    c[0] = particleSet[0]->weight;
    for (int i = 1; i < this->getNumberOfParticles(); ++i)
	{
        c[i] = c[i-1] + particleSet[i]->weight;
    }

    // stochastic universal (systematic) resampling
	// step size
	const double step = 1.0 / static_cast<double>(this->numberOfParticles);
    //initialize starting position u
    double u = Util::uniformRandom(0.0, step);

    vector<Particle*> newSet;
    newSet.reserve(this->numberOfParticles);

    //index
    int idx = 0;

    // resample particles
    for (int i = 0; i < this->getNumberOfParticles(); ++i) {
        // Increment Threshold
        double threshold = u + i * step;
        
        //skip until next threshold reached
        while (threshold > c[idx] && idx < this->getNumberOfParticles() - 1 ){
			idx++;
		}
        
    	//add new sampled particle
        // newSet.push_back(new Particle(this->particleSet[idx]->x, this->particleSet[idx]->y, this->particleSet[idx]->theta, 1.0/(double)this->numberOfParticles));
		Particle* np = new Particle(*particleSet[idx]);
        np->weight = log(step); // reset to log uniform weight for next iteration
        newSet.push_back(np);
	}

	// delete old particles and replace with new set
    for (Particle* p : this->particleSet)
		delete p;
	this->particleSet = newSet;
}

Particle* ParticleFilter::getBestHypothesis() {
	return this->bestHypothesis;
}

// added for convenience
int ParticleFilter::computeMapIndex(int width, int height, int x,
		int y) {
	return x + y * width;
}

