#include <stdlib.h>
#include <string>

#include <fstream>
#include <iostream>
#include <list>
#include <dirent.h>
#include <algorithm>
#include <numeric>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "include/brisk/brisk.h"
#include "VisualFeatureEncoding.h"
#include "VisualFeatureDecoding.h"
#include "VisualFeatureEncodingInter.h"
#include "VisualFeatureDecodingInter.h"
#include <stdio.h>
#include <ctime>
#include <functional>


#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <bitset>

// #include "utils.h"

using namespace cv;
using namespace std;


int main(int argc, char ** argv) {


	// VisualFeatureDecoding decoder;

	VisualFeatureEncodingInter enc_inter;
	VisualFeatureDecodingInter dec_inter;
	VisualFeatureEncoding enc_intra;

	vector< int >           num_feats;
	Ptr<FeatureDetector>     det;
	Ptr<DescriptorExtractor> ext;


	vector<KeyPoint> cur_kpts, ref_kpts;
	Mat cur_frame, ref_frame, cur_f_kpts, ref_f_kpts, cur_frame_rgb;
	Mat cur_desc, ref_desc;

	BRISK_pModel *pModel = BRISK_pModel::get_instance();
	BRISKinter_pModel *interpModel = BRISKinter_pModel::get_instance();

	InterCodingInfo enc_info;

	int NFRAME = 100;


	//LOAD VIDEO
	// string filename = "images/pompadour.mov";
	string filename = "images/hall.mp4";
	// string filename = "images/foreman.mp4";
	VideoCapture vid(filename);

	if( !vid.isOpened() )
	        throw "Error when reading steam_avi";

	//CREATE BRISK DETECTOR
	det  = new cv::BriskFeatureDetector(60 ,4);


	//LOAD OPTIMAL PAIRS
	vector<int> pairs;
	pModel->getPairs(pairs);


    vector<int> relative_pairs;
    interpModel->getRelativePairs(relative_pairs);


	//CREATE BRISK DESCRIPTOR
	ext = new cv::BriskDescriptorExtractor(pairs);

	//INIT 1st FRAME
	vid >> cur_frame_rgb;

	Size im_size(cur_frame_rgb.cols, cur_frame_rgb.rows);

	cvtColor(cur_frame_rgb,cur_frame,CV_RGB2GRAY);

	//DETECT AND COMPUTE KEYPOINTS
	det->detect(cur_frame, cur_kpts);
	ext->compute(cur_frame, cur_kpts, cur_desc);

	for (int frame = 1; frame < NFRAME; frame++){

		vector< uchar > kpts_bitstream;
		vector< uchar > desc_bitstream_fullintra;
		vector< uchar > desc_bitstream_intra;
		vector< uchar > mv_bitstream;
		vector< uchar > refIDs_bitstream;
		vector< uchar > desc_bitstream_inter;

		// cur IS THE NEW ref
		ref_frame = cur_frame;
		ref_kpts = cur_kpts;
		ref_desc = cur_desc;

		//RETRIEVE FRAME AND COMPUTE kpts AND desc
		vid >> cur_frame_rgb;
		cvtColor(cur_frame_rgb,cur_frame,CV_RGB2GRAY);

		det->detect(cur_frame, cur_kpts);

	    ext->compute(cur_frame, cur_kpts, cur_desc);

	    cv::drawKeypoints(cur_frame, cur_kpts, cur_f_kpts);
	    cv::drawKeypoints(ref_frame, ref_kpts, ref_f_kpts);


	    //ENCODE DATA INTRA
	    clock_t t0 = clock();
	    enc_intra.encodeBinaryDescriptors("BRISK", cur_desc, desc_bitstream_fullintra);
	    clock_t t1 = clock() - t0;


	    //COMPUTE GAIN INTRA
	    cout << "Uncompressed size: " << cur_desc.rows*cur_desc.cols << endl;
	    cout << "Intra Compressed size: " << desc_bitstream_fullintra.size() << endl;
	    cout << "Coding gain: " << (float)100*(cur_desc.rows*cur_desc.cols - desc_bitstream_fullintra.size())/((cur_desc.rows*cur_desc.cols)) << "%" << endl;
	    cout << "Execution time: " << (((float) t1)/CLOCKS_PER_SEC)*1000.0 << " ms" << endl;


	    vector<int> cur_IDs;

	    //ENCODE DATA INTER
	    t0 = clock();
	    enc_inter.encodeBinaryFeaturesInter("BRISK", ref_kpts, cur_kpts, ref_desc, cur_desc, kpts_bitstream,
	    									desc_bitstream_intra, mv_bitstream, refIDs_bitstream,
	    		                            desc_bitstream_inter, false, im_size, enc_info, cur_IDs);
	    t1 = clock() - t0;

	    cout << endl;

	    //COMPUTE GAIN INTER
	    cout << "Uncompressed size: " << cur_desc.rows*cur_desc.cols << endl;
	    cout  << "Inter Compressed size: " << desc_bitstream_inter.size() << endl;
	    cout  << "Intra Compressed size: " << desc_bitstream_intra.size() << endl;
	    cout  << "Compressed size: " << desc_bitstream_intra.size() + desc_bitstream_inter.size() << endl;
	    cout << "Coding gain: " << (float)100*(cur_desc.rows*cur_desc.cols - (desc_bitstream_inter.size() + desc_bitstream_intra.size()))/((cur_desc.rows*cur_desc.cols)) << "%" << endl;
	    cout << "Execution time: " << (((float) t1)/CLOCKS_PER_SEC)*1000.0 << " ms" << endl;



	    cout << enc_info;
	    cout << "Execution time: " << (((float) t1)/CLOCKS_PER_SEC)*1000.0 << " ms" << endl;

		vector<int> decoded_refIDs;
	    dec_inter.decodeIDs(refIDs_bitstream, decoded_refIDs, enc_info.getFeatInter());

	    vector<KeyPoint> decoded_inter_kpts;
	    dec_inter.decodeKeypointsFromMVs(mv_bitstream, ref_kpts, decoded_refIDs, decoded_inter_kpts);

	    if (VERBOSE){
	    	for (unsigned int cc = 0; cc < cur_IDs.size(); cc++){
	    		cout << "[x = " << cur_kpts[cur_IDs[cc]].pt.x << ", y = " << cur_kpts[cur_IDs[cc]].pt.y << "] - [x = " << decoded_inter_kpts[cc].pt.x << ", y = " << decoded_inter_kpts[cc].pt.y << "]" << endl;
	    	}
	    }

	    Mat cur_desc_inter;
	    dec_inter.decodeBinaryDescriptorsFromPRs("BRISK", desc_bitstream_inter, ref_desc, decoded_refIDs, cur_desc_inter);

	    if (VERBOSE){
	    	for (unsigned int cc = 0; cc < cur_IDs.size(); cc++){
	    		cout << "original descriptor: " << endl;
	    		for (int dd = 0; dd < cur_desc_inter.cols; dd++){
	    			cout <<  (unsigned) cur_desc.at<uchar>(cur_IDs[cc], dd) << " ";
	    		}
	    		cout << endl << "decoded descriptor: " << endl;
	    		for (int dd = 0; dd < cur_desc_inter.cols; dd++){
	    			cout << (unsigned) cur_desc_inter.at<uchar>(cc, dd) << " ";
	    		}
	    		cout << endl;

	    	}
	    }

	    imshow("Reference frame", ref_f_kpts);
	    imshow("Current frame", cur_f_kpts);
	    cout << endl << "Press a key for next frame..." << endl;
	    cv::waitKey();

	}
}

