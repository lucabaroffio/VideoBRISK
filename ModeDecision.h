/*
 * ModeDecision.h
 *
 *  Created on: 29/gen/2014
 *      Author: lucabaroffio
 */

#ifndef MODEDECISION_H_
#define MODEDECISION_H_

#include <iostream>
#include <stdio.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"

using namespace cv;
using namespace std;

class ModeDecision {

	double lambda_;

public:
	ModeDecision();
	ModeDecision(double lambda_);
	virtual ~ModeDecision();

	double getLambda() const {
		return lambda_;
	}

	void setLambda(double lambda) {
		lambda_ = lambda;
	}

	int BRISKModeDecision_DescOnly(Mat ref_desc, Mat cur_desc, vector<int> out_IDs,
								   vector<int> &cur_inter_IDs, vector<int> &ref_inter_IDs, vector<int> &cur_intra_IDs);

	int BinaryModeDecision_DistThresh(Mat ref_desc, Mat cur_desc, vector<int> out_IDs,
									  vector<int> &cur_inter_IDs, vector<int> &ref_inter_IDs, vector<int> &cur_intra_IDs, double thresh, vector<double> &dists);

	int BRISKModeDecision(Mat ref_desc, Mat cur_desc, vector<KeyPoint> ref_kpts, vector<KeyPoint> cur_kpts, vector<int> out_IDs,
										vector<int> &cur_inter_IDs, vector<int> &ref_inter_IDs, vector<int> &ref_intra_IDs, cv::Size im_size);

};

#endif /* MODEDECISION_H_ */
