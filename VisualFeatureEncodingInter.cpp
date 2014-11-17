/*
 * VisualFeatureEncoding.cpp
 */
#include "VisualFeatureEncodingInter.h"
#include "VisualFeatureEncoding.h"
#include "MotionEstimation.h"
#include "ModeDecision.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "assert.h"

#include "global.h"

// bool VERBOSE = true;

VisualFeatureEncodingInter::VisualFeatureEncodingInter(){

}

int VisualFeatureEncodingInter::encodeMotionVectors(vector<KeyPoint> ref_kpts, vector<KeyPoint> cur_kpts, vector<int> cur_IDs, vector<int> ref_IDs,
						vector<uchar>& bitstream){

	// load model for motion vectors
	motionvectors_pModel *mvpModel = motionvectors_pModel::get_instance();

	// cout << "START ENCODING..." << endl;
	ac_encoder ace; // the encoder
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


	ac_encoder_init (&ace, bitstream); // init the encoder

	int center_bin_x = (int) MVXSIZE/2 + 1;
	int center_bin_y = (int) MVYSIZE/2 + 1;
	int center_bin_s = (int) MVSSIZE/2 + 1;

	for(unsigned int cc_idx=0; cc_idx < cur_IDs.size(); cc_idx++){

		int cc = cur_IDs[cc_idx];

		// get the current motion vector
		double cur_mv_x = cur_kpts[cc].pt.x - ref_kpts[ref_IDs[cc_idx]].pt.x;
		double cur_mv_y = cur_kpts[cc].pt.y - ref_kpts[ref_IDs[cc_idx]].pt.y;
		double cur_mv_s = cur_kpts[cc].size - ref_kpts[ref_IDs[cc_idx]].size;


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


		ac_model_init (&acm, MVXSIZE, NULL, 0);  // init the model
		ac_encode_symbol_updateModel(&ace, &acm, cur_bin_x, freq_x);

		ac_model_init (&acm, MVYSIZE, NULL, 0);  // init the model
		ac_encode_symbol_updateModel(&ace, &acm, cur_bin_y, freq_y);

		// encode scale
		// ac_model_init (&acm, MVSSIZE, NULL, 0);  // init the model
		// ac_encode_symbol_updateModel(&ace, &acm, cur_bin_s, freq_s);

		// compute rate
		//				double cur_mv_cost = - log2(mvpModel->getP_mv_x(cur_bin_x))
		//										 - log2(mvpModel->getP_mv_y(cur_bin_y))
		//										 - log2(mvpModel->getP_mv_s(cur_bin_s));

	}

	ac_encoder_done (&ace);
	ac_model_done (&acm);

	return 0;
}

int VisualFeatureEncodingInter::encodePredictionResidualBRISK(Mat ref_desc, Mat cur_desc, vector<int> cur_IDs, vector<int> ref_IDs, vector<uchar>& bitstream){

	// load the BRISK probability model
	BRISKinter_pModel *pModelInter = BRISKinter_pModel::get_instance();

	// cout << "START ENCODING..." << endl;
	ac_encoder ace; // the encoder
	ac_model   acm; // the probability model used by ace

	int freq[2]; // vector of frequency (dynamically updated)

	ac_encoder_init (&ace, bitstream); // init the encoder
	ac_model_init (&acm, 2, NULL, 0);  // init the model

	int cur_bit, prev_bit;

	double temp_p;

	// start coding

	for(unsigned int cc_idx=0; cc_idx < cur_IDs.size(); cc_idx++){

		int cc = cur_IDs[cc_idx];

		int bin_res[BRISK_LENGTH_BITS];
		for (int dd = 0; dd < cur_desc.cols; dd++){
			uchar cur_el = cur_desc.at<uchar>(cc,dd);
			uchar ref_el = ref_desc.at<uchar>(ref_IDs[cc_idx],dd);
			uchar resid = cur_el^ref_el;

			for(int j=0;j<8;j++){
				bin_res[dd*8 + (7- j)] = (resid>>j) & 0x0001;
			}
		}

		int index=0;


		for (int dd_bin = 0; dd_bin < BRISK_LENGTH_BITS; dd_bin++){

			prev_bit = cur_bit;
			cur_bit = bin_res[pModelInter->get_inter_relative_idx(dd_bin) - 1];// (cur_el >> j) & 0x0001;
			// read the probabilities

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

			ac_encode_symbol_updateModel(&ace, &acm, cur_bit, freq);

			index++;
		}
	}

	ac_encoder_done (&ace);
	ac_model_done (&acm);

	return 0;
}

int VisualFeatureEncodingInter::encodeIDs(vector<int> &cur_IDs, vector<int> &ref_IDs, vector<uchar>& bitstream){

	double p_id = 1/MAX_FEAT;

	ac_encoder ace; // the encoder
	ac_model   acm; // the probability model used by ace

	int freq[MAX_FEAT]; // vector of frequency (dynamically updated)

	for (int ff = 0; ff < MAX_FEAT; ff++){
		freq[ff] = AC_PRECISION * p_id + 1;
	}

	ac_encoder_init (&ace, bitstream); // init the encoder
	ac_model_init (&acm, MAX_FEAT, NULL, 0);  // init the model

	for (unsigned int rr = 0; rr < ref_IDs.size(); rr++){

		assert(ref_IDs[rr] < MAX_FEAT);
		ac_encode_symbol_updateModel(&ace, &acm, ref_IDs[rr], freq);

	}

	ac_encoder_done (&ace);
	ac_model_done (&acm);

	return 0;
}


int VisualFeatureEncodingInter::encodeBinaryFeaturesInter(string descName, vector<KeyPoint> ref_kpts, vector<KeyPoint> cur_kpts,
							Mat ref_desc, Mat cur_desc, vector<uchar>& kpts_bitstream,
							vector<uchar>& desc_bitstream_intra, vector<uchar>& mv_bitstream, vector<uchar>& IDs_bitstream,
							vector<uchar>& desc_bitstream_inter, bool descOnly, Size im_size, InterCodingInfo& enc_info,
							vector<int> &out_cur_IDs){

	MotionEstimation *me = new MotionEstimation();
	ModeDecision *md = new ModeDecision();

	// vector<int> out_candidates[cur_kpts.size()];
	vector<vector<int> > out_candidates;

	vector<int> inter_cur_IDs;
	vector<int> inter_ref_IDs;

	vector<int> intra_cur_IDs;

	vector<double> inter_dists;

	vector<int> refIDs;


	vector<double> dists;

	// candidate selection
	me->CandidateSelection(ref_kpts, cur_kpts, out_candidates);

	//DEBUG
	for (unsigned int cc = 0; cc < cur_kpts.size(); cc++){
		if (VERBOSE){
			cout << "feature " << cc+1 << endl;
			for (unsigned int jj=0; jj < out_candidates[cc].size(); jj++){
				cout << out_candidates[cc][jj] << " ";
			}
			cout << endl;
		}
	}

	// motion estimation
	// SIMPLE MOTION ESTIMATION, J = D
	me->BinaryMotionEstimation_DistOnly(ref_kpts, cur_kpts, ref_desc, cur_desc, out_candidates, refIDs, dists);

	// COMPLETE MOTION ESTIMATION, J = D + \lambda R^{mv}
	// me->BinaryMotionEstimation(ref_kpts, cur_kpts, ref_desc, cur_desc, out_candidates, refIDs, dists);

	// COMPLETE MOTION ESTIMATION W/O CANDIDATE SELECTION
	// me->BinaryMotionEstimation(ref_kpts, cur_kpts, ref_desc, cur_desc, refIDs, dists);

	if (VERBOSE){
		for (unsigned int cc = 0; cc < cur_kpts.size(); cc++){
			cout << "MATCHING CANDIDATE FEATURE, current feature # " << cc <<":" << endl;
			cout << refIDs[cc] << ", ";
			cout << "D = " << dists[cc];
			cout << endl;
		}
	}

	// mode decision
	if (strcmp(descName.c_str(), "BRISK") == 0){

		// COMPLETE MODE DECISION, J_inter = R_inter, J_intra = R_intra
		md->BRISKModeDecision(ref_desc, cur_desc, ref_kpts, cur_kpts, refIDs, inter_cur_IDs, inter_ref_IDs, intra_cur_IDs, im_size);

		// SIMPLE MODE DECISION, if (D_inter < thresh) then INTER else INTRA
		// md->BinaryModeDecision_DistThresh(ref_desc, cur_desc, refIDs, inter_cur_IDs, inter_ref_IDs, intra_cur_IDs, 0.3, dists);

		if (VERBOSE){
			for (unsigned int cc = 0; cc < inter_cur_IDs.size(); cc++){
				cout << "MATCHING FEATs: [" << inter_cur_IDs[cc] << ", " << inter_ref_IDs[cc] << "] "<< endl;
				cout << endl;
			}

			cout << "NON-MATCHING FEATs:" << endl << "{";
			for (unsigned int cc = 0; cc < intra_cur_IDs.size(); cc++){
				cout << intra_cur_IDs[cc] << ", ";
			}
			cout << "}" << endl;
		}
	}

	Mat cur_desc_intra(intra_cur_IDs.size(), cur_desc.cols, CV_8U);
	for (unsigned int cc = 0; cc < intra_cur_IDs.size(); cc++){
		for (int dd = 0; dd < cur_desc.cols; dd++){
		cur_desc_intra.at<uchar>(cc, dd) = cur_desc.at<uchar>(intra_cur_IDs[cc], dd);
		}
	}

	// encode intra
	VisualFeatureEncoding encoder;
	encoder.encodeBinaryDescriptors(descName, cur_desc_intra, desc_bitstream_intra);

	// encode inter
	if (strcmp(descName.c_str(), "BRISK") == 0){
		encodePredictionResidualBRISK(ref_desc, cur_desc, inter_cur_IDs, inter_ref_IDs, desc_bitstream_inter);
	}

	// encode reference keypoint id
	encodeIDs(inter_cur_IDs, inter_ref_IDs, IDs_bitstream);

	if (!descOnly){

		// encode intra location
		vector<KeyPoint> intra_kpts;
		for (unsigned int ia = 0; ia < intra_cur_IDs.size(); ia++){
			intra_kpts.push_back(cur_kpts[ia]);
		}

		encoder.encodeKeyPoints(intra_kpts, kpts_bitstream, im_size.width, im_size.height);

		// encode inter motion vectors
		encodeMotionVectors(ref_kpts, cur_kpts, inter_cur_IDs, inter_ref_IDs, mv_bitstream);
		out_cur_IDs = inter_cur_IDs;

	}

	enc_info.setFeatIntra(cur_desc_intra.rows);
	enc_info.setFeatInter(inter_cur_IDs.size());
	enc_info.setRateInter(desc_bitstream_inter.size() + mv_bitstream.size() + IDs_bitstream.size());
	enc_info.setRateIntra(desc_bitstream_intra.size() + kpts_bitstream.size());
	enc_info.setBytesPerEntryInter(((double) desc_bitstream_inter.size())/((double) BRISK_LENGTH_BITS)/((double) inter_cur_IDs.size()));
	enc_info.setBytesPerEntryIntra(((double) desc_bitstream_intra.size())/((double) BRISK_LENGTH_BITS)/((double) cur_desc_intra.rows));
	enc_info.setBytesPerFeatureGlobal(((double) enc_info.getRateInter() + enc_info.getRateIntra())/((double) enc_info.getFeatInter() + enc_info.getFeatIntra()));
	enc_info.setBytesPerIdInter(((double) IDs_bitstream.size())/((double) inter_cur_IDs.size()));
	enc_info.setBytesPerLocationInter(((double) mv_bitstream.size())/((double) inter_cur_IDs.size()));
	enc_info.setBytesPerLocationIntra(((double) kpts_bitstream.size())/((double) cur_desc_intra.rows));
	enc_info.setRateUncompressed(cur_desc.rows*cur_desc.cols + cur_kpts.size()*log2(((double) im_size.width)/PRECX) + cur_kpts.size()*log2(((double) im_size.height)/PRECY));
	enc_info.setCodingGain(((double) enc_info.getRateUncompressed() - (double) enc_info.getRateInter() - (double) enc_info.getRateIntra())/((double) enc_info.getRateUncompressed()));
	enc_info.setCodingGainDesc(((double) cur_desc.rows*cur_desc.cols - (double) desc_bitstream_inter.size() - (double) desc_bitstream_intra.size())/((double) cur_desc.rows*cur_desc.cols));

	return 0;
}



