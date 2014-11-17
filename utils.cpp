/*
 * utils.cpp
 *
 *  Created on: 03/feb/2014
 *      Author: lucabaroffio
 */

#include "utils.h"

#include <bitset>
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "CodecParams.h"
#include "include/brisk/brisk.h"


using namespace std;
using namespace cv;

size_t hamming(bitset<8> a, bitset<8> b){



        return (a^b).count();

}


double hammingD(Mat a, Mat b){


	double hD = 0;
	for (int dd = 0; dd < a.cols; dd++){

		bitset<8> op1(a.at<char>(0, dd));
		bitset<8> op2(b.at<char>(0, dd));

		hD+=hamming(op1, op2);

	}

	return hD/(a.cols * 8);
}

int readVideo_saveBRISK(char* in_path, char* out_path, int N_FRAME,
		int thresh, int octaves){

//	char* in_path = "/Users/lucabaroffio/Movies/paris/paris.mp4";
//	char* out_path = "/Users/lucabaroffio/Movies/paris/new_BRISK";
//	int N_FRAME = 299;

	Mat frame, descs, frame2;

	Ptr<FeatureDetector> det = new cv::BriskFeatureDetector(thresh,octaves);
	Ptr<DescriptorExtractor> ext;
	std::vector<cv::KeyPoint> kpts;
	Mat descriptors;

	BRISK_pModel *pModel = BRISK_pModel::get_instance();

	vector<int> pairs;
	pModel->getPairs(pairs);

	//CREATE BRISK DESCRIPTOR
	ext = new cv::BriskDescriptorExtractor(pairs);


	char o_path1[200] = "";
	//        strcpy(o_path1, path);
	strcat(o_path1, out_path);
	strcat(o_path1, "/D.bin");
	ofstream fileD(o_path1, ios::out|ios::binary);
	cout << o_path1 << endl;

	char o_path2[200]= "";
	//        strcpy(o_path2, path);
	strcat(o_path2, out_path);
	strcat(o_path2, "/F.bin");
	ofstream fileF(o_path2, ios::out|ios::binary);
	cout << o_path2 << endl;

	char o_path3[200] = "";
	//        strcpy(o_path3, path);
	strcat(o_path3, out_path);
	strcat(o_path3, "/N.bin");
	ofstream fileN(o_path3, ios::out|ios::binary);
	cout << o_path3 << endl;

	//        ofstream fileF(strcat(o_path, "/F.bin"), ios::out|ios::binary);
	//        ofstream fileN(strcat(o_path, "/N.bin"), ios::out|ios::binary);



	cout << in_path << endl;


	VideoCapture v(in_path);

	if ( !v.isOpened() )  // if not success, exit program
	{
		cout << "Cannot open the video file" << endl;
		return -1;
	}

	for (int f = 0; f < N_FRAME; f++){

		//            cout << "reading frame " << f << endl;

		v.read(frame);

		//            cout << "frame  " << f << "read" << endl;


		det->detect(frame, kpts);

		//            if (strcmp(desc,"FREAK")==0){
		//                for (int kk = 0; kk < kpts.size(); kk++){
		//                    kpts.at(kk).size = kpts.at(kk).size*4;
		//                }
		//
		//            }

		for (unsigned int i = 0; i < kpts.size(); i++){
			kpts.at(i).size = kpts.at(i).size*1;
		}

		ext->compute(frame, kpts, descs);

		//            drawKeypoints(frame, kpts, frame2);
		//            imshow("MyVideo", frame2); //show the frame in "MyVideo" window
		//            waitKey();
		cout << f << " ";

		if (fileN.is_open())
		{
			fileN.write(reinterpret_cast<char *>(&descs.rows), sizeof(descs.rows));

		}

		else{
			cout << "Unable to open file fileN";
			return 100;
		}

		for (int dd = 0; dd < descs.rows; dd++){

			if (fileF.is_open())
			{

				fileF.write(reinterpret_cast<char *>(&kpts[dd].pt.x), sizeof(float));
				fileF.write(reinterpret_cast<char *>(&kpts[dd].pt.y), sizeof(float));
				fileF.write(reinterpret_cast<char *>(&kpts[dd].size), sizeof(float));
			}

			else{
				cout << "Unable to open file fileF";
				return 200;
			}

			for (int dex = 0; dex < descs.cols; dex++){


				if (fileD.is_open())
				{
					fileD.write(reinterpret_cast<char *>(&descs.at<char>(dd, dex)), sizeof(char));
				}

				else{
					cout << "Unable to open file fileD";
					return 300;
				}



			}
		}

		cout << endl;

	}
	return 0;
}


