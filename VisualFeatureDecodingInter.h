/*
 * VisualFeatureDecodingInter.h
 *
 *  Created on: 12/feb/2014
 *      Author: lucabaroffio
 */

#ifndef VISUALFEATUREDECODINGINTER_H_
#define VISUALFEATUREDECODINGINTER_H_

#include <iostream>
#include <stdio.h>
#include <math.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "thirdparty/coder/ac_extended.h"

#include "CodecParams.h"

using namespace cv;
using namespace std;

class VisualFeatureDecodingInter {
public:
	int decodeKeypointsFromMVs(vector<uchar> bitstream, vector<KeyPoint> ref_kpts, vector<int> ref_IDs, vector<KeyPoint> &cur_kpts);
	int decodeBinaryDescriptorsFromPRs(string descName, vector<uchar> bitstream, Mat ref_desc, vector<int> ref_IDs, Mat &cur_desc);
	int decodeIDs(vector<uchar> bitstream, vector<int> &ref_IDs, int Nfeats);


	int decodeNonBinaryDescriptors(Mat features, vector<uchar>& bitstream);


	VisualFeatureDecodingInter();
	virtual ~VisualFeatureDecodingInter();

private:

	int decodeBRISK_PRs(vector<uchar> bitstream, Mat ref_desc, vector<int> ref_IDs, Mat &cur_desc);
};

#endif /* VISUALFEATUREDECODINGINTER_H_ */
