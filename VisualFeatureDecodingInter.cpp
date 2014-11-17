/*
 * VisualFeatureDecodingInter.cpp
 *
 *  Created on: 12/feb/2014
 *      Author: lucabaroffio
 */

#include "VisualFeatureDecodingInter.h"

VisualFeatureDecodingInter::VisualFeatureDecodingInter() {
	// TODO Auto-generated constructor stub

}

VisualFeatureDecodingInter::~VisualFeatureDecodingInter() {
	// TODO Auto-generated destructor stub
}

int VisualFeatureDecodingInter::decodeIDs(vector<uchar> bitstream, vector<int> &ref_IDs, int Nfeats){

		double p_id = 1/MAX_FEAT;

		ac_decoder acd; // the encoder
		ac_model   acm; // the probability model used by ace

		int freq[MAX_FEAT]; // vector of frequency (dynamically updated)

		for (int ff = 0; ff < MAX_FEAT; ff++){
			freq[ff] = AC_PRECISION * p_id + 1;
		}

		ac_decoder_init (&acd, bitstream); // init the encoder
		ac_model_init (&acm, MAX_FEAT, NULL, 0);  // init the model


		for (int cc = 0; cc < Nfeats; cc++){

				ref_IDs.push_back(ac_decode_symbol_updateModel(&acd, &acm, freq));
		}

		ac_decoder_done (&acd);
		ac_model_done (&acm);

		return 0;
}

int VisualFeatureDecodingInter::decodeKeypointsFromMVs(vector<uchar> bitstream, vector<KeyPoint> ref_kpts, vector<int> ref_IDs, vector<KeyPoint> &cur_kpts){

	// load model for motion vectors
	motionvectors_pModel *mvpModel = motionvectors_pModel::get_instance();

	// cout << "START ENCODING..." << endl;
	ac_decoder acd; // the encoder
	ac_model   acm; // the probability model used by ace

	int freq_x[MVXSIZE];
	int freq_y[MVYSIZE];
	int freq_s[MVSSIZE];


	for (int i = 0; i < MVXSIZE; i++){
		freq_x[i] = AC_PRECISION*mvpModel->getP_mv_x(i) + 1;
	}

	for (int i = 0; i < MVYSIZE; i++){
		freq_y[i] = AC_PRECISION*mvpModel->getP_mv_y(i) + 1;
	}

	for (int i = 0; i < MVSSIZE; i++){
		freq_s[i] = AC_PRECISION*mvpModel->getP_mv_s(i) + 1;
	}


	ac_decoder_init (&acd, bitstream); // init the encoder

	int center_bin_x = (int) MVXSIZE/2 + 1;
	int center_bin_y = (int) MVYSIZE/2 + 1;
	// int center_bin_s = (int) MVSSIZE/2 + 1;

	for(unsigned int cc_idx=0; cc_idx < ref_IDs.size(); cc_idx++){

		ac_model_init (&acm, MVXSIZE, NULL, 0);  // init the model
	    int cur_bin_x = ac_decode_symbol_updateModel(&acd, &acm, freq_x);

	    ac_model_init (&acm, MVYSIZE, NULL, 0);  // init the model
	    int cur_bin_y = ac_decode_symbol_updateModel(&acd, &acm, freq_y);

	    //DECODE THE SCALE PARAMETER
	    // ac_model_init (&acm, MVSSIZE, NULL, 0);  // init the model
	    // int cur_bin_s = ac_decode_symbol_updateModel(&acd, &acm, freq_s);


		// get the right bin of the LookUp Table
		double cur_mv_x = (cur_bin_x - center_bin_x)*PRECX;
		double cur_mv_y = (cur_bin_y - center_bin_y)*PRECY;
		// get the right scale
		// double cur_mv_s = (cur_bin_s - center_bin_s)*PRECS;

		// get the current motion vector
		KeyPoint* temp_kp = new KeyPoint(cur_mv_x + ref_kpts[ref_IDs[cc_idx]].pt.x, cur_mv_y + ref_kpts[ref_IDs[cc_idx]].pt.y, 0, 0, 0, 0, 0);
		cur_kpts.push_back(*temp_kp);


	}

	ac_decoder_done (&acd);
	ac_model_done (&acm);

	return 0;

}


int VisualFeatureDecodingInter::decodeBRISK_PRs(vector<uchar> bitstream, Mat ref_desc, vector<int> ref_IDs, Mat &cur_desc){

	// load the BRISK probability model
	BRISKinter_pModel *pModelInter = BRISKinter_pModel::get_instance();

	cur_desc = *(new Mat(ref_IDs.size(), ref_desc.cols, CV_8U));

	// cout << "START ENCODING..." << endl;
	ac_decoder acd; // the encoder
	ac_model   acm; // the probability model used by ace

	int freq[2]; // vector of frequency (dynamically updated)

	ac_decoder_init (&acd, bitstream); // init the encoder
	ac_model_init (&acm, 2, NULL, 0);  // init the model

	int cur_bit, prev_bit;

	double temp_p;

	// start coding

	for(unsigned int cc_idx=0; cc_idx < ref_IDs.size(); cc_idx++){

		int bin_res[BRISK_LENGTH_BITS];


		for (int dd_bin = 0; dd_bin < BRISK_LENGTH_BITS; dd_bin++){

			prev_bit = cur_bit;

			if( dd_bin == 0){
				temp_p = min(pModelInter->getP0(pModelInter->get_inter_idx(dd_bin) - 1), 0.99);

				freq[0] = (int)max(1, (int)round( temp_p * (double)AC_PRECISION ) );
				freq[1] = AC_PRECISION - freq[0];

			}
			else{
				if( prev_bit == 0){
					temp_p = max(min(pModelInter->getP0c0(pModelInter->get_inter_idx(dd_bin) - 1, pModelInter->get_inter_idx(dd_bin - 1) - 1), 0.99), 0.01);
					if (isnan(temp_p))
						temp_p = 0.85;

					freq[0] = (int)max(1, (int)round( temp_p * (double)AC_PRECISION ) );
					freq[1] = AC_PRECISION - freq[0];
				}
				else{
					temp_p = max(min(pModelInter->getP0c1(pModelInter->get_inter_idx(dd_bin) - 1, pModelInter->get_inter_idx(dd_bin - 1) - 1), 0.99), 0.01);
					if (isnan(temp_p))
						temp_p = 0.50;

					freq[0] = (int)max(1, (int)round( temp_p * (double)AC_PRECISION ) );
					freq[1] = AC_PRECISION - freq[0];
				}
			}

			if (freq[0]==AC_PRECISION){
				freq[0] = AC_PRECISION-1;
				freq[1] = 1;
			}

//			freq[0] = 0.9*AC_PRECISION;
//			freq[1] = 0.1*AC_PRECISION;

			cur_bit = ac_decode_symbol_updateModel(&acd, &acm, freq);
			bin_res[pModelInter->get_inter_relative_idx(dd_bin) - 1] = cur_bit;

		}

		// add (XOR) the residual and wrote it back
		for (int dd = 0; dd < ref_desc.cols; dd++){

			uchar cur_resid = 0;
			for(int j=0;j<8;j++){
				cur_resid |= ((uchar) (bin_res[8*dd + 7 - j] << j));
			}
			cur_desc.at<uchar>(cc_idx,dd) = ref_desc.at<uchar>(ref_IDs[cc_idx], dd)^cur_resid;
		}
	}

	ac_decoder_done (&acd);
	ac_model_done (&acm);

	return 0;

}


int VisualFeatureDecodingInter::decodeBinaryDescriptorsFromPRs(string descName, vector<uchar> bitstream, Mat ref_desc, vector<int> ref_IDs, Mat &cur_desc){

	if (strcmp(descName.c_str(), "BRISK") == 0){
		return decodeBRISK_PRs(bitstream, ref_desc, ref_IDs, cur_desc);
	}
	else return -2;
}



