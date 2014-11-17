/*
 * MotionEstimation.cpp
 *
 *  Created on: 27/gen/2014
 *      Author: lucabaroffio
 */

#include "MotionEstimation.h"
#include "utils.h"
#include "global.h"

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;


MotionEstimation::MotionEstimation(float lambda, int deltaX, int deltaY, int deltaS, int k){



		this->lambda_ = lambda;
		this->deltaS_ = deltaS;
		this->deltaX_ = deltaX;
		this->deltaY_ = deltaY;
		this->k_ = k;

}

// default constructor
MotionEstimation::MotionEstimation(){



		this->lambda_ = 1;
		this->deltaS_ = 10;
		this->deltaX_ = 30;
		this->deltaY_ = 30;
		this->k_ = 5;

}

MotionEstimation::~MotionEstimation() {
	// TODO Auto-generated destructor stub
}

// int MotionEstimation::CandidateSelection(vector<KeyPoint> ref_kpts, vector<KeyPoint> cur_kpts, vector<int> out_candidates[]){

int MotionEstimation::CandidateSelection(vector<KeyPoint> ref_kpts, vector<KeyPoint> cur_kpts, vector<vector<int> > &out_candidates){

	// out_candidates.reserve(cur_kpts.size());
	KeyPoint kk;

	// for each kpt of current frame
	for (unsigned int cc = 0; cc < cur_kpts.size(); cc++){

		float cur_x = cur_kpts[cc].pt.x;
		float cur_y = cur_kpts[cc].pt.y;
		float cur_s = cur_kpts[cc].size;

		vector<int> temp;

		// check whether reference kpt is inside the spatial window
		for (unsigned int rr = 0; rr < ref_kpts.size(); rr++){

			if ((ref_kpts[rr].pt.x > cur_x - deltaX_)&&(ref_kpts[rr].pt.x < cur_x + deltaX_)){
				if ((ref_kpts[rr].pt.y > cur_y - deltaY_)&&(ref_kpts[rr].pt.y < cur_y + deltaY_)){
					if ((ref_kpts[rr].size > cur_s - deltaS_)&&(ref_kpts[rr].size < cur_s + deltaS_)){

						temp.push_back(rr);

						// out_candidates[cc].push_back(rr); // add to the vector of candidates
					}
				}
			}
		}
		out_candidates.push_back(temp);
	}

	return 0;
}




//int MotionEstimation::BinaryMotionEstimation(vector<KeyPoint> ref_kpts, vector<KeyPoint> cur_kpts,
//				Mat ref_desc, Mat cur_desc, vector<int> mask[], vector<int> &out_IDs, vector<double> &dists){

int MotionEstimation::BinaryMotionEstimation(vector<KeyPoint> ref_kpts, vector<KeyPoint> cur_kpts,
				Mat ref_desc, Mat cur_desc, vector<vector<int> > mask, vector<int> &out_IDs, vector<double> &dists){

	dists.clear();
	motionvectors_pModel *mvpModel = motionvectors_pModel::get_instance();
	double x_prec = mvpModel->getPrecX();
	double y_prec = mvpModel->getPrecY();
	double s_prec = mvpModel->getPrecS();

	int center_bin_x = (int) MVXSIZE/2 + 1;
	int center_bin_y = (int) MVYSIZE/2 + 1;
	int center_bin_s = (int) MVSSIZE/2 + 1;

	if (VERBOSE)
		cout << center_bin_x << " " << center_bin_y << endl << endl;


	// for each current descriptor
	for (unsigned int cc = 0; cc < cur_kpts.size(); cc++){

		if (mask[cc].size()>0){

			if (VERBOSE){
				cout << "feature " << cc << endl;
			}

			vector<pair<double, int> > J_cur;
			vector<pair<double, int> > D_cur;
			vector<int> cur_cand = mask[cc];
			// for each element in the mask vector

			for (unsigned int cand = 0; cand < mask[cc].size(); cand++){

				// compute hamming distance
				D_cur.push_back(make_pair(hammingD(cur_desc.row(cc), ref_desc.row(cur_cand[cand])), cur_cand[cand]));
			}



			// find the k-nearest neighbor
			sort(D_cur.begin(), D_cur.end());


			for (unsigned int tbc = 0; tbc < D_cur.size(); tbc++){
				// compute the costs of top-k motion vectors

				// check only the top-k candidates according to distortion
				if (tbc >= k_)
					break;

				// get the current motion vector
				double cur_mv_x = cur_kpts[cc].pt.x - ref_kpts[D_cur[tbc].second].pt.x;
				double cur_mv_y = cur_kpts[cc].pt.y - ref_kpts[D_cur[tbc].second].pt.y;
				double cur_mv_s = cur_kpts[cc].size - ref_kpts[D_cur[tbc].second].size;


				// get the right bin of the LookUp Table
				int cur_bin_x = (int) center_bin_x + floor(cur_mv_x/x_prec);
				cur_bin_x = max(0, cur_bin_x);
				cur_bin_x = min(MVXSIZE, cur_bin_x);

				int cur_bin_y = (int) center_bin_y + floor(cur_mv_y/y_prec);
				cur_bin_y = max(0, cur_bin_y);
				cur_bin_y = min(MVYSIZE, cur_bin_y);

				int cur_bin_s = (int) center_bin_s + floor(cur_mv_s/s_prec);
				cur_bin_s = max(0, cur_bin_s);
				cur_bin_s = min(MVSSIZE, cur_bin_s);


				// compute rate
//				double cur_mv_cost = - log2(mvpModel->getP_mv_x(cur_bin_x))
//										 - log2(mvpModel->getP_mv_y(cur_bin_y))
//										 - log2(mvpModel->getP_mv_s(cur_bin_s));

				double cur_mv_cost = - log2(mvpModel->getP_mv_x(cur_bin_x))
									 - log2(mvpModel->getP_mv_y(cur_bin_y));


				//DEBUG
				if (VERBOSE){
						cout << "mv_x = " << cur_mv_x << ", cur_bin_x = " << cur_bin_x << ", cost_x = " << - log2(mvpModel->getP_mv_x(cur_bin_x)) << endl;
						cout << "mv_y = " << cur_mv_y << ", cur_bin_y = " << cur_bin_y << ", cost_y = " << - log2(mvpModel->getP_mv_y(cur_bin_y)) << endl;
						cout << "mv_s = " << cur_mv_s << ", cur_bin_s = " << cur_bin_s << ", cost_s = " << - log2(mvpModel->getP_mv_s(cur_bin_s)) << endl;

						cout << "[D = " << D_cur[tbc].first << ", R = " << cur_mv_cost << ", id = " << D_cur[tbc].second << "] " << endl;
				}


				// compute cost function, J = D + lambda R^{mv}
				J_cur.push_back(make_pair(D_cur[tbc].first + lambda_*cur_mv_cost, D_cur[tbc].second));
			}

			if (VERBOSE){
				cout << endl;
			}

			// reference feature is the one that minimizes J
			pair<double, int> min_J = *min_element(J_cur.begin(), J_cur.end());
			out_IDs.push_back(min_J.second);
			for (unsigned int tbc = 0; tbc < D_cur.size(); tbc++){
				if (D_cur[tbc].second == min_J.second)
					dists.push_back(D_cur[tbc].first);
			}


		}
		else{
			out_IDs.push_back(-1);
		    dists.push_back(-1);
		}
	}

	return 0;
}

int MotionEstimation::BinaryMotionEstimation(vector<KeyPoint> ref_kpts, vector<KeyPoint> cur_kpts,
				Mat ref_desc, Mat cur_desc, vector<int> &out_IDs, vector<double> &dists){

	// vector<int> mask[cur_desc.rows];
	vector<vector<int> > mask;
	mask.reserve(cur_desc.rows);

	for (int cc = 0; cc < cur_desc.rows; cc++){
		for (int rr = 0; rr < ref_desc.rows; rr++){
			mask[cc].push_back(rr);
		}
	}
	return this->BinaryMotionEstimation(ref_kpts, cur_kpts, ref_desc, cur_desc, mask, out_IDs, dists);

}

//int MotionEstimation::BinaryMotionEstimation_DistOnly(vector<KeyPoint> ref_kpts, vector<KeyPoint> cur_kpts,
//				Mat ref_desc, Mat cur_desc, vector<int> mask[], vector<int> &out_IDs, vector<double> &dists){

int MotionEstimation::BinaryMotionEstimation_DistOnly(vector<KeyPoint> ref_kpts, vector<KeyPoint> cur_kpts,
				Mat ref_desc, Mat cur_desc, vector<vector<int> > mask, vector<int> &out_IDs, vector<double> &dists){

	out_IDs.clear();
	dists.clear();

	// for each current descriptor
	for (unsigned int cc = 0; cc < cur_kpts.size(); cc++){

		vector<pair<double, int> > D_cur;
		vector<int> cur_cand = mask[cc];
		// for each element in the mask vector
		for (unsigned int cand = 0; cand < mask[cc].size(); cand++){

			// compute hamming distance
			double hd = hammingD(cur_desc.row(cc), ref_desc.row(cur_cand[cand]));
			D_cur.push_back(make_pair(hd, cur_cand[cand]));
		}

		//DEBUG
		if (VERBOSE){
			cout << "feature " << cc << endl;
			for (unsigned int jj = 0; jj<D_cur.size(); jj++){
				cout << "[" << D_cur[jj].first << ", " << D_cur[jj].second << "] ";
			}
			cout << endl;
		}

		if (mask[cc].size()>0){
			// find the k-nearest neighbor
			pair<double, int> cur_match = *min_element(D_cur.begin(), D_cur.end());
			out_IDs.push_back(cur_match.second);
			dists.push_back(cur_match.first);
		}
		else{
			out_IDs.push_back(-1);
			dists.push_back(-1);
		}

	}

	return 0;
}




