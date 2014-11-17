/*
 * ModeDecision.cpp
 *
 *  Created on: 29/gen/2014
 *      Author: lucabaroffio
 */

#include "ModeDecision.h"
#include "CodecParams.h"
#include "utils.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"

using namespace std;
using namespace cv;

ModeDecision::ModeDecision() {
	// TODO Auto-generated constructor stub
	lambda_ = 1;
}

ModeDecision::ModeDecision(double lambda) {
	// TODO Auto-generated constructor stub
	lambda_ = lambda;
}

ModeDecision::~ModeDecision() {
	// TODO Auto-generated destructor stub
}

int ModeDecision::BinaryModeDecision_DistThresh(Mat ref_desc, Mat cur_desc, vector<int> out_IDs,
												vector<int> &cur_inter_IDs, vector<int> &ref_inter_IDs, vector<int> &cur_intra_IDs, double thresh, vector<double> &dists){

	for (unsigned int cc = 0; cc < out_IDs.size(); cc++){
		if (out_IDs[cc] >= 0){
			double d = hammingD(cur_desc.row(cc), ref_desc.row(out_IDs[cc]));
			if (d<thresh){
				cur_inter_IDs.push_back(cc);
				ref_inter_IDs.push_back(out_IDs[cc]);
				dists.push_back(d);
			}
			else{
				cur_intra_IDs.push_back(cc);
			}
		}
	}
	return 0;
}



int ModeDecision::BRISKModeDecision_DescOnly(Mat ref_desc, Mat cur_desc, vector<int> out_IDs,
											 vector<int> &cur_inter_IDs, vector<int> &ref_inter_IDs, vector<int> &cur_intra_IDs){

	// load model for inter
	BRISKinter_pModel *pModelInter = BRISKinter_pModel::get_instance();

	// load model for intra
	BRISK_pModel *pModelIntra = BRISK_pModel::get_instance();

	// load model for motion vectors
	// motionvectors_pModel *pModelMv = motionvectors_pModel::get_instance();

	for (int cc = 0; cc < cur_desc.rows; cc++){

		int index=0;
		int cur_bit, prev_bit;

		double intra_rate = 0;
		double inter_rate = 0;

		double temp_p;



		if (out_IDs[cc]>=0){

			// compute cost intra
			for (int dd = 0; dd < cur_desc.cols; dd++){
				// get the current element
				uchar cur_el = cur_desc.at<uchar>(cc,dd);

				//convert it into binary
				int temp_el[8];

				for(int j=0;j<8;j++){
					temp_el[7-j] = (cur_el>>j) & 0x0001;
				}

				for(int j=0;j<8;j++){

					prev_bit = cur_bit;
					cur_bit = temp_el[j];// (cur_el >> j) & 0x0001;
					// read the probabilities
					if( index == 0){
						temp_p = pModelIntra->getP0(index);
						intra_rate = (cur_bit == 0 ? intra_rate - log2(temp_p) : intra_rate - log2(1.0 - temp_p));

					}
					else{
						if( prev_bit == 0){
							temp_p = pModelIntra->getP0c0(index, index - 1);
							intra_rate = (cur_bit == 0 ? intra_rate - log2(temp_p) : intra_rate - log2(1.0 - temp_p));
						}
						else{
							temp_p = pModelIntra->getP0c1(index, index - 1);
							intra_rate = (cur_bit == 0 ? intra_rate - log2(temp_p) : intra_rate - log2(1.0 - temp_p));
						}
					}

					index++;

				}

			}

			// compute cost inter

			int bin_res[BRISK_LENGTH_BITS];
			for (int dd = 0; dd < cur_desc.cols; dd++){
				uchar cur_el = cur_desc.at<uchar>(cc,dd);
				uchar ref_el = ref_desc.at<uchar>(out_IDs[cc],dd);
				uchar resid = cur_el^ref_el;

				for(int j=0;j<8;j++){
					bin_res[dd*8 + (7- j)] = (resid>>j) & 0x0001;
				}
			}

			for (int dd_bin = 0; dd_bin < BRISK_LENGTH_BITS; dd_bin++){

					prev_bit = cur_bit;
					cur_bit = bin_res[pModelInter->get_inter_relative_idx(dd_bin) - 1];// (cur_el >> j) & 0x0001;
					// read the probabilities

					if( dd_bin == 0){
						temp_p = min(pModelInter->getP0(pModelInter->get_inter_idx(dd_bin) - 1), 0.99);
						inter_rate = (cur_bit == 0 ? inter_rate - log2(temp_p) : inter_rate - log2(1.0 - temp_p));

					}
					else{
						if( prev_bit == 0){
							temp_p = max(min(pModelInter->getP0c0(pModelInter->get_inter_idx(dd_bin) - 1, pModelInter->get_inter_idx(dd_bin - 1) - 1), 0.99), 0.01);
							if (isnan(temp_p))
								temp_p = 0.85;
							inter_rate = (cur_bit == 0 ? inter_rate - log2(temp_p) : inter_rate - log2(1.0 - temp_p));
						}
						else{
							temp_p = max(min(pModelInter->getP0c1(pModelInter->get_inter_idx(dd_bin) - 1, pModelInter->get_inter_idx(dd_bin - 1) - 1), 0.99), 0.01);
							if (isnan(temp_p))
								temp_p = 0.50;
							inter_rate = (cur_bit == 0 ? inter_rate - log2(temp_p) : inter_rate - log2(1.0 - temp_p));
						}
					}
			}
			// compare cost functions
			double J_inter = inter_rate;
			double J_intra = intra_rate;

			// choose between intra and inter
			if (J_inter < J_intra){
				cur_inter_IDs.push_back(cc);
				ref_inter_IDs.push_back(out_IDs[cc]);
			}
			else{
				cur_intra_IDs.push_back(cc);
			}

		}
		else{
			cur_intra_IDs.push_back(cc);
		}
	}

	return 0;
}


int ModeDecision::BRISKModeDecision(Mat ref_desc, Mat cur_desc, vector<KeyPoint> ref_kpts, vector<KeyPoint> cur_kpts, vector<int> out_IDs,
									vector<int> &cur_inter_IDs, vector<int> &ref_inter_IDs, vector<int> &cur_intra_IDs, cv::Size im_size){




	// load model for inter
	BRISKinter_pModel *pModelInter = BRISKinter_pModel::get_instance();

	// load model for intra
	BRISK_pModel *pModelIntra = BRISK_pModel::get_instance();

	// load model for motion vectors
	motionvectors_pModel *mvpModel = motionvectors_pModel::get_instance();



	int center_bin_x = (int) MVXSIZE/2 + 1;
	int center_bin_y = (int) MVYSIZE/2 + 1;
	int center_bin_s = (int) MVSSIZE/2 + 1;

	double intra_kpts_rate = -log2(1/(im_size.height*mvpModel->getPrecY())) -log2(1/(im_size.width*mvpModel->getPrecX()));

	for (int cc = 0; cc < cur_desc.rows; cc++){

		int index=0;
		int cur_bit, prev_bit;

		double intra_rate = 0;
		double inter_rate = 0;

		double inter_kpts_rate;

		double temp_p;



		if (out_IDs[cc]>=0){

			// compute cost intra
			for (int dd = 0; dd < cur_desc.cols; dd++){
				// get the current element
				uchar cur_el = cur_desc.at<uchar>(cc,dd);

				//convert it into binary
				int temp_el[8];

				for(int j=0;j<8;j++){
					temp_el[7-j] = (cur_el>>j) & 0x0001;
				}

				for(int j=0;j<8;j++){

					prev_bit = cur_bit;
					cur_bit = temp_el[j];// (cur_el >> j) & 0x0001;
					// read the probabilities
					if( index == 0){
						temp_p = pModelIntra->getP0(index);
						intra_rate = (cur_bit == 0 ? intra_rate - log2(temp_p) : intra_rate - log2(1.0 - temp_p));

					}
					else{
						if( prev_bit == 0){
							temp_p = pModelIntra->getP0c0(index, index - 1);
							intra_rate = (cur_bit == 0 ? intra_rate - log2(temp_p) : intra_rate - log2(1.0 - temp_p));
						}
						else{
							temp_p = pModelIntra->getP0c1(index, index - 1);
							intra_rate = (cur_bit == 0 ? intra_rate - log2(temp_p) : intra_rate - log2(1.0 - temp_p));
						}
					}

					index++;

				}

			}

			// compute cost inter

			int bin_res[BRISK_LENGTH_BITS];
			for (int dd = 0; dd < cur_desc.cols; dd++){
				uchar cur_el = cur_desc.at<uchar>(cc,dd);
				uchar ref_el = ref_desc.at<uchar>(out_IDs[cc],dd);
				uchar resid = cur_el^ref_el;

				for(int j=0;j<8;j++){
					bin_res[dd*8 + (7- j)] = (resid>>j) & 0x0001;
				}
			}

			for (int dd_bin = 0; dd_bin < BRISK_LENGTH_BITS; dd_bin++){

					prev_bit = cur_bit;
					cur_bit = bin_res[pModelInter->get_inter_relative_idx(dd_bin) - 1];// (cur_el >> j) & 0x0001;
					// read the probabilities

					if( dd_bin == 0){
						temp_p = min(pModelInter->getP0(pModelInter->get_inter_idx(dd_bin) - 1), 0.99);
						inter_rate = (cur_bit == 0 ? inter_rate - log2(temp_p) : inter_rate - log2(1.0 - temp_p));

					}
					else{
						if( prev_bit == 0){
							temp_p = max(min(pModelInter->getP0c0(pModelInter->get_inter_idx(dd_bin) - 1, pModelInter->get_inter_idx(dd_bin - 1) - 1), 0.99), 0.01);
							if (isnan(temp_p))
								temp_p = 0.85;
							inter_rate = (cur_bit == 0 ? inter_rate - log2(temp_p) : inter_rate - log2(1.0 - temp_p));
						}
						else{
							temp_p = max(min(pModelInter->getP0c1(pModelInter->get_inter_idx(dd_bin) - 1, pModelInter->get_inter_idx(dd_bin - 1) - 1), 0.99), 0.01);
							if (isnan(temp_p))
								temp_p = 0.50;
							inter_rate = (cur_bit == 0 ? inter_rate - log2(temp_p) : inter_rate - log2(1.0 - temp_p));
						}
					}
			}

			// compute cost for motion vectors

			// get the current motion vector
			double cur_mv_x = cur_kpts[cc].pt.x - ref_kpts[out_IDs[cc]].pt.x;
			double cur_mv_y = cur_kpts[cc].pt.y - ref_kpts[out_IDs[cc]].pt.y;
			double cur_mv_s = cur_kpts[cc].size - ref_kpts[out_IDs[cc]].size;


			// get the right bin of the LookUp Table
			int cur_bin_x = (int) center_bin_x + floor(cur_mv_x/PRECX);
			cur_bin_x = max(0, cur_bin_x);
			cur_bin_x = min(MVXSIZE - 1, cur_bin_x);

			int cur_bin_y = (int) center_bin_y + floor(cur_mv_y/PRECY);
			cur_bin_y = max(0, cur_bin_y);
			cur_bin_y = min(MVYSIZE - 1, cur_bin_y);

			int cur_bin_s = (int) center_bin_s + floor(cur_mv_s/PRECS);
			cur_bin_s = max(0, cur_bin_s);
			cur_bin_s = min(MVSSIZE - 1, cur_bin_s);


			// compute rate
			//				double cur_mv_cost = - log2(mvpModel->getP_mv_x(cur_bin_x))
			//										 - log2(mvpModel->getP_mv_y(cur_bin_y))
			//										 - log2(mvpModel->getP_mv_s(cur_bin_s));

			inter_kpts_rate = - log2(mvpModel->getP_mv_x(cur_bin_x))
							  - log2(mvpModel->getP_mv_y(cur_bin_y));


			// compare cost functions
			double J_inter = inter_rate + inter_kpts_rate;
			double J_intra = intra_rate + intra_kpts_rate;

			// choose between intra and inter
			if (J_inter < J_intra){
				cur_inter_IDs.push_back(cc);
				ref_inter_IDs.push_back(out_IDs[cc]);
			}
			else{
				cur_intra_IDs.push_back(cc);
			}

		}
		else{
			cur_intra_IDs.push_back(cc);
		}
	}



	return 0;
}


