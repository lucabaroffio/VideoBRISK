/*
 * MotionEstimation.h
 *
 *  Created on: 27/gen/2014
 *      Author: lucabaroffio
 */

#ifndef MOTIONESTIMATION_H_
#define MOTIONESTIMATION_H_


#include <iostream>
#include <stdio.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"

using namespace cv;
using namespace std;


class MotionEstimation {



	float lambda_;
	int deltaX_;
	int deltaY_;
	int deltaS_;
	unsigned int k_; // rate of motion vector is computed only for top-k features in terms of distortion

public:

	MotionEstimation();

	MotionEstimation(float, int, int, int, int);

	virtual ~MotionEstimation();

	// Motion estimation w/o candidate mask
//	int BinaryMotionEstimation(vector<KeyPoint> ref_kpts, vector<KeyPoint> cur_kpts,
//			Mat ref_desc, Mat cur_desc, vector<int> &out_IDs, vector<double> &dists);
//
//	// Motion estimation w/ candidate mask
//	int BinaryMotionEstimation(vector<KeyPoint> ref_kpts, vector<KeyPoint> cur_kpts,
//				Mat ref_desc, Mat cur_desc, vector<int> mask[], vector<int> &out_IDs, vector<double> &dists);

	int BinaryMotionEstimation(vector<KeyPoint> ref_kpts, vector<KeyPoint> cur_kpts,
			Mat ref_desc, Mat cur_desc, vector<int> &out_IDs, vector<double> &dists);

	// Motion estimation w/ candidate mask
	int BinaryMotionEstimation(vector<KeyPoint> ref_kpts, vector<KeyPoint> cur_kpts,
				Mat ref_desc, Mat cur_desc, vector<vector<int> > mask, vector<int> &out_IDs, vector<double> &dists);

	int CandidateSelection(vector<KeyPoint> ref_kpts, vector<KeyPoint> cur_kpts, vector<vector<int> > &out_candidates);


	int BinaryMotionEstimation_DistOnly(vector<KeyPoint> ref_kpts, vector<KeyPoint> cur_kpts,
					Mat ref_desc, Mat cur_desc, vector<vector<int> > mask, vector<int> &out_IDs, vector<double> &dists);



	int getDeltaS() const {
		return deltaS_;
	}

	void setDeltaS(int deltaS) {
		deltaS_ = deltaS;
	}

	int getDeltaX() const {
		return deltaX_;
	}

	void setDeltaX(int deltaX) {
		deltaX_ = deltaX;
	}

	int getDeltaY() const {
		return deltaY_;
	}

	void setDeltaY(int deltaY) {
		deltaY_ = deltaY;
	}

	unsigned int getK() const {
		return k_;
	}

	void setK(unsigned int k) {
		k_ = k;
	}

	float getLambda() const {
		return lambda_;
	}

	void setLambda(float lambda) {
		lambda_ = lambda;
	}
};

#endif /* MOTIONESTIMATION_H_ */
