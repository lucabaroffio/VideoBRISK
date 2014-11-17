/*
 * utils.h
 *
 *  Created on: 28/gen/2014
 *      Author: lucabaroffio
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <bitset>
#include "opencv2/core/core.hpp"
#include "CodecParams.h"
#include "include/brisk/brisk.h"


using namespace std;
using namespace cv;

size_t hamming(bitset<8> a, bitset<8> b);


double hammingD(Mat a, Mat b);

int readVideo_saveBRISK(char* in_path, char* out_path, int N_FRAME,
		int thresh, int octaves);



#endif /* UTILS_H_ */
