/*
 * VisualFeatureEncodingInter.h
 */
#ifndef VISUALFEATUREENCODINGINTER_H_
#define VISUALFEATUREENCODINGINTER_H_

#include <iostream>
#include <stdio.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "thirdparty/coder/ac_extended.h"

#include "CodecParams.h"

using namespace cv;
using namespace std;


class InterCodingInfo{

	private:
		double n_feat_intra;
		double n_feat_inter;
		double bytes_per_entry_intra;
		double bytes_per_location_intra;
		double bytes_per_entry_inter;
		double bytes_per_location_inter;
		double bytes_per_ID_inter;
		double bytes_per_feature_global;
		double rate_inter;
		double rate_intra;
		double rate_uncompressed;
		double coding_gain;
		double coding_gain_desc;

	public:

		void print(){
			cout << "Number of intra-coded features: " << n_feat_intra << endl;
			cout << "Number of inter-coded features: " << n_feat_inter << endl;
			cout << "Bits per entry, intra-coded features: " << bytes_per_entry_intra*8 << endl;
			cout << "Bits per location, intra-coded features: " << bytes_per_location_intra*8 << endl;
			cout << "Bits per entry, inter-coded features: " << bytes_per_entry_inter*8 << endl;
			cout << "Bites per location, inter-coded features: " << bytes_per_location_inter*8 << endl;
			cout << "Bits per reference keypoint id, inter-coded features: " << bytes_per_ID_inter*8 << endl;
			cout << "Bits per feature, global: " << bytes_per_feature_global*8 << endl;
			cout << "Rate intra: " << rate_intra << " Bytes" <<  endl;
			cout << "Rate inter: " << rate_inter << " Bytes" <<  endl;
			cout << "Rate uncompressed: " << rate_uncompressed << " Bytes" <<  endl;
			cout << "Coding gain: " << coding_gain*100 << "%" <<  endl;
			cout << "Coding gain, descriptor only: " << coding_gain_desc*100 << "%" <<  endl;
		}

		friend ostream& operator<<(ostream& os, const InterCodingInfo& ci){

				os << "Number of intra-coded features: " << ci.n_feat_intra << endl;
				os << "Number of inter-coded features: " << ci.n_feat_inter << endl;
				os << "Bits per entry, intra-coded features: " << ci.bytes_per_entry_intra*8 << endl;
				os << "Bits per location, intra-coded features: " << ci.bytes_per_location_intra*8 << endl;
				os << "Bits per entry, inter-coded features: " << ci.bytes_per_entry_inter*8 << endl;
				os << "Bites per location, inter-coded features: " << ci.bytes_per_location_inter*8 << endl;
				os << "Bits per reference keypoint id, inter-coded features: " << ci.bytes_per_ID_inter*8 << endl;
				os << "Bits per feature, global: " << ci.bytes_per_feature_global*8 << endl;
				os << "Rate intra: " << ci.rate_intra << " Bytes" <<  endl;
				os << "Rate inter: " << ci.rate_inter << " Bytes" <<  endl;
				os << "Rate uncompressed: " << ci.rate_uncompressed << " Bytes" <<  endl;
				os << "Coding gain: " << ci.coding_gain*100 << "%" <<  endl;
				os << "Coding gain, descriptor only: " << ci.coding_gain_desc*100 << "%" <<  endl;

				return os;
		}


		double getBytesPerEntryInter() const {
			return bytes_per_entry_inter;
		}

		void setBytesPerEntryInter(double bytesPerEntryInter) {
			bytes_per_entry_inter = bytesPerEntryInter;
		}

		double getBytesPerEntryIntra() const {
			return bytes_per_entry_intra;
		}

		void setBytesPerEntryIntra(double bytesPerEntryIntra) {
			bytes_per_entry_intra = bytesPerEntryIntra;
		}

		double getBytesPerFeatureGlobal() const {
			return bytes_per_feature_global;
		}

		void setBytesPerFeatureGlobal(double bytesPerFeatureGlobal) {
			bytes_per_feature_global = bytesPerFeatureGlobal;
		}

		double getBytesPerIdInter() const {
			return bytes_per_ID_inter;
		}

		void setBytesPerIdInter(double bytesPerIdInter) {
			bytes_per_ID_inter = bytesPerIdInter;
		}

		double getBytesPerLocationInter() const {
			return bytes_per_location_inter;
		}

		void setBytesPerLocationInter(double bytesPerLocationInter) {
			bytes_per_location_inter = bytesPerLocationInter;
		}

		double getBytesPerLocationIntra() const {
			return bytes_per_location_intra;
		}

		void setBytesPerLocationIntra(double bytesPerLocationIntra) {
			bytes_per_location_intra = bytesPerLocationIntra;
		}

		double getFeatInter() const {
			return n_feat_inter;
		}

		void setFeatInter(double featInter) {
			n_feat_inter = featInter;
		}

		double getFeatIntra() const {
			return n_feat_intra;
		}

		void setFeatIntra(double featIntra) {
			n_feat_intra = featIntra;
		}

	double getRateInter() const {
		return rate_inter;
	}

	void setRateInter(double rateInter) {
		rate_inter = rateInter;
	}

	double getRateIntra() const {
		return rate_intra;
	}

	void setRateIntra(double rateIntra) {
		rate_intra = rateIntra;
	}

	double getCodingGain() const {
		return coding_gain;
	}

	void setCodingGain(double codingGain) {
		coding_gain = codingGain;
	}

	double getRateUncompressed() const {
		return rate_uncompressed;
	}

	void setRateUncompressed(double rateUncompressed) {
		rate_uncompressed = rateUncompressed;
	}

	double getCodingGainDesc() const {
		return coding_gain_desc;
	}

	void setCodingGainDesc(double codingGainDesc) {
		coding_gain_desc = codingGainDesc;
	}
};

class VisualFeatureEncodingInter
{

public:

	VisualFeatureEncodingInter();
	int encodeBinaryFeaturesInter(string descName, vector<KeyPoint> ref_kpts, vector<KeyPoint> cur_kpts,
								Mat ref_desc, Mat cur_desc, vector<uchar>& kpts_bitstream,
								vector<uchar>& desc_bitstream_intra, vector<uchar>& mv_bitstream, vector<uchar>& IDs_bitstream,
								vector<uchar>& desc_bitstream_inter, bool descOnly, Size im_size, InterCodingInfo &enc_info, vector<int> &inter_cur_IDs);

	int dummy_encodeBinaryFeaturesInter(string descName, vector<KeyPoint> ref_kpts, vector<KeyPoint> cur_kpts,
								Mat ref_desc, Mat cur_desc, vector<uchar>& kpts_bitstream,
								vector<uchar>& desc_bitstream_intra, vector<uchar>& mv_bitstream, vector<uchar>& IDs_bitstream,
								vector<uchar>& desc_bitstream_inter, bool descOnly, Size im_size);

private:

	int encodePredictionResidualBRISK(Mat ref_desc, Mat cur_desc, vector<int> cur_IDs, vector<int> ref_IDs, vector<uchar>& bitstream);

	int encodeMotionVectors(vector<KeyPoint> ref_kpts, vector<KeyPoint> cur_kpts, vector<int> cur_IDs, vector<int> ref_IDs,
						vector<uchar>& bitstream);

	int encodeIDs(vector<int> &cur_IDs, vector<int> &ref_IDs, vector<uchar>& bitstream);


};





#endif /* VISUALFEATUREENCODINGINTER_H_ */
