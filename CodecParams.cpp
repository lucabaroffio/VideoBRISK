#include "CodecParams.h"
#include <math.h>


BRISK_pModel* BRISK_pModel::get_instance(){

	if (instance_ptr == null) {
		instance_ptr = new BRISK_pModel();
	}
	return instance_ptr;

}

BRISK_pModel::BRISK_pModel(){
	loadFiles();
}

double BRISK_pModel::getP0(int id){
	return P0[ idxs[id]-1];
};

double BRISK_pModel::getP0c0(int id1, int id2){
	return P0c0[ idxs[id1]-1 ][ idxs[id2]-1 ];
};

double BRISK_pModel::getP0c1(int id1, int id2){
	return P0c1[ idxs[id1]-1 ][ idxs[id2]-1 ];
};


void BRISK_pModel::loadFiles(){

	cout << "LOADING THE FILES!" << endl;
	// load the probability model
	ifstream file;

	file.open("thirdparty/coder/brisk/P0.bin", ios::in | ios::binary);
	file.read((char*)P0,sizeof(double)*1776);
	file.close();
//	cout << "P0" << endl;
//	for (int i=0;i<1776;i++)
//		cout << P0[i] << endl;

	file.open("thirdparty/coder/brisk/P0c0.bin", ios::in | ios::binary);
	for(int i=0; i<1776; i++){
		for(int j=0; j<1776; j++){
			file.read((char*)&P0c0[j][i],sizeof(double));
		}
	}
	file.close();

	file.open("thirdparty/coder/brisk/P0c1.bin", ios::in | ios::binary);
	for(int i=0; i<1776; i++){
		for(int j=0; j<1776; j++){
			file.read((char*)&P0c1[j][i],sizeof(double));
		}
	}
	file.close();

	switch(BRISK_LENGTH_BITS){
	case 512:
		file.open("thirdparty/coder/brisk/ranking_original_optimized512.bin", ios::in | ios::binary);
		break;
	case 256:
		file.open("thirdparty/coder/brisk/ranking_original_optimized256.bin", ios::in | ios::binary);
		break;
	case 128:
		file.open("thirdparty/coder/brisk/ranking_original_optimized128.bin", ios::in | ios::binary);
		break;
	case 112:
		file.open("thirdparty/coder/brisk/inter/relative_ranking_inter_original_optimized112.bin", ios::in | ios::binary);
		break;
	case 96:
		file.open("thirdparty/coder/brisk/ranking_original_optimized96.bin", ios::in | ios::binary);
		break;
	case 64:
		file.open("thirdparty/coder/brisk/ranking_original_optimized64.bin", ios::in | ios::binary);
		break;
	default:
		file.open("thirdparty/coder/brisk/ranking_original_optimized512.bin", ios::in | ios::binary);
		break;
	}
	file.read((char*)idxs, sizeof(int)*BRISK_LENGTH_BITS);
	file.close();
	cout << "INTRA FILES LOADED!" << endl;
}

int BRISK_pModel::getPairs(vector<int> &pairs){
	for(int i = 0; i<BRISK_LENGTH_BITS; i++){
		pairs.push_back(idxs[i]);
		cout << idxs[i] << " ";
	}
	return 0;
}

BRISK_pModel* BRISK_pModel::instance_ptr = null;


//////////////////////////////////////////////////////////////////////
// INTER MODEL
//////////////////////////////////////////////////////////////////////

BRISKinter_pModel* BRISKinter_pModel::get_instance(){

	if (inter_instance_ptr == null) {
		inter_instance_ptr = new BRISKinter_pModel();
	}
	return inter_instance_ptr;

}

BRISKinter_pModel::BRISKinter_pModel(){
	this->loadFiles();
}

double BRISKinter_pModel::getP0(int id){
	return P0[id - 1];
};

double BRISKinter_pModel::getP0c0(int id1, int id2){
	return P0c0[ id1 - 1 ][ id2 - 1];
};

double BRISKinter_pModel::getP0c1(int id1, int id2){
	return P0c1[ id1 - 1 ][ id2 - 1 ];
};

int BRISKinter_pModel::get_inter_idx(int id){
	return inter_idxs[id];
}

int BRISKinter_pModel::get_inter_relative_idx(int id){
	return relative_idxs[id];
}


void BRISKinter_pModel::loadFiles(){

	cout << "LOADING THE FILES!" << endl;
	// load the probability model
	ifstream file;

	file.open("thirdparty/coder/brisk/inter/P0.bin", ios::in | ios::binary);
	file.read((char*)P0,sizeof(double)*1776);
	file.close();
//	cout << "P0" << endl;
//	for (int i=0;i<1776;i++)
//		cout << P0[i] << endl;

	file.open("thirdparty/coder/brisk/inter/P0c0.bin", ios::in | ios::binary);
	for(int i=0; i<1776; i++){
		for(int j=0; j<1776; j++){
			file.read((char*)&P0c0[j][i],sizeof(double));
		}
	}
	file.close();

	file.open("thirdparty/coder/brisk/inter/P0c1.bin", ios::in | ios::binary);
	for(int i=0; i<1776; i++){
		for(int j=0; j<1776; j++){
			file.read((char*)&P0c1[j][i],sizeof(double));
		}
	}
	file.close();

	switch(BRISK_LENGTH_BITS){
	case 512:
		file.open("thirdparty/coder/brisk/inter/ranking_inter_original_optimized512.bin", ios::in | ios::binary);
		break;
	case 256:
		file.open("thirdparty/coder/brisk/inter/ranking_inter_original_optimized256.bin", ios::in | ios::binary);
		break;
	case 128:
		file.open("thirdparty/coder/brisk/inter/ranking_inter_original_optimized128.bin", ios::in | ios::binary);
		break;
	case 112:
		file.open("thirdparty/coder/brisk/inter/relative_ranking_inter_original_optimized112.bin", ios::in | ios::binary);
		break;
	case 96:
		file.open("thirdparty/coder/brisk/inter/ranking_inter_original_optimized96.bin", ios::in | ios::binary);
		break;
	case 64:
		file.open("thirdparty/coder/brisk/inter/ranking_inter_original_optimized64.bin", ios::in | ios::binary);
		break;
	default:
		file.open("thirdparty/coder/brisk/inter/ranking_inter_original_optimized128.bin", ios::in | ios::binary);
		break;
	}

	file.read((char*) this->inter_idxs, sizeof(int)*BRISK_LENGTH_BITS);
	file.close();

	switch(BRISK_LENGTH_BITS){
		case 512:
			file.open("thirdparty/coder/brisk/inter/relative_ranking_inter_original_optimized512.bin", ios::in | ios::binary);
			break;
		case 256:
			file.open("thirdparty/coder/brisk/inter/relative_ranking_inter_original_optimized256.bin", ios::in | ios::binary);
			break;
		case 128:
			file.open("thirdparty/coder/brisk/inter/relative_ranking_inter_original_optimized128.bin", ios::in | ios::binary);
			break;
		case 112:
			file.open("thirdparty/coder/brisk/inter/relative_ranking_inter_original_optimized112.bin", ios::in | ios::binary);
			break;
		case 96:
			file.open("thirdparty/coder/brisk/inter/relative_ranking_inter_original_optimized96.bin", ios::in | ios::binary);
			break;
		case 64:
			file.open("thirdparty/coder/brisk/inter/relative_ranking_inter_original_optimized64.bin", ios::in | ios::binary);
			break;
		default:
			file.open("thirdparty/coder/brisk/inter/relative_ranking_inter_original_optimized128.bin", ios::in | ios::binary);
			break;
	}

	file.read((char*) this->relative_idxs, sizeof(int)*BRISK_LENGTH_BITS);
	file.close();
	cout << "INTER FILES LOADED!" << endl;
}

int BRISKinter_pModel::getPairs(vector<int> &pairs){
	for(int i = 0; i<BRISK_LENGTH_BITS; i++){
		pairs.push_back(inter_idxs[i]);
		cout << inter_idxs[i] << " ";
	}
	return 0;
}

int BRISKinter_pModel::getRelativePairs(vector<int> &pairs){
	for(int i = 0; i<BRISK_LENGTH_BITS; i++){
		pairs.push_back(relative_idxs[i]);
	}
	return 0;
}

BRISKinter_pModel* BRISKinter_pModel::inter_instance_ptr = null;



//////////////////////////////////////////////////////////////////////
// MOTION VECTORS MODEL
//////////////////////////////////////////////////////////////////////

motionvectors_pModel* motionvectors_pModel::get_instance(){

	if (mv_instance_ptr == null) {
		mv_instance_ptr = new motionvectors_pModel();
	}
	return mv_instance_ptr;

}

double motionvectors_pModel::getP_mv_x(int id1){
	return this->p_mv_x[id1];
}

double motionvectors_pModel::getP_mv_y(int id1){
	return this->p_mv_y[id1];
}

double motionvectors_pModel::getP_mv_s(int id1){
	return this->p_mv_s[id1];
}

double* motionvectors_pModel::getP_mv_x(){

	return this->p_mv_x;
}

double* motionvectors_pModel::getP_mv_y(){

	return this->p_mv_y;
}

double* motionvectors_pModel::getP_mv_s(){

	return this->p_mv_s;
}

double motionvectors_pModel::getP_mv_x(double val){
	int center_bin = MVXSIZE/2 + 1;
	//int bin_shift = (int) round(val/this->PREC_x);
	int bin_shift = (int) round(val/PRECX);
	return this->p_mv_x[center_bin + bin_shift];
}


double motionvectors_pModel::getP_mv_y(double val){
	int center_bin = MVYSIZE/2 + 1;
	//int bin_shift = (int) round(val/this->PREC_y);
	int bin_shift = (int) round(val/PRECY);
	return this->p_mv_y[center_bin + bin_shift];
}

double motionvectors_pModel::getP_mv_s(double val){
	int center_bin = MVYSIZE/2 + 1;
	//int bin_shift = (int) round(val/this->PREC_s);
	int bin_shift = (int) round(val/PRECS);
	return this->p_mv_s[center_bin + bin_shift];
}

motionvectors_pModel::motionvectors_pModel(){
	this->loadFiles();
}


void motionvectors_pModel::loadFiles(){

	cout << "LOADING THE FILES!" << endl;
	// load the probability model
	ifstream file;


	file.open("thirdparty/coder/brisk/inter/p_mv_x.bin", ios::in | ios::binary);
	file.read((char*) &p_mv_x, sizeof(double)*MVXSIZE);
	file.close();

	file.open("thirdparty/coder/brisk/inter/p_mv_y.bin", ios::in | ios::binary);
	file.read((char*) &p_mv_y, sizeof(double)*MVYSIZE);
	file.close();

	file.open("thirdparty/coder/brisk/inter/p_mv_s.bin", ios::in | ios::binary);
	file.read((char*) &p_mv_s, sizeof(double)*MVSSIZE);
	file.close();

	cout << "MOTION VECTOR FILES LOADED!" << endl;
}



motionvectors_pModel* motionvectors_pModel::mv_instance_ptr = null;


//////////////////////////////////////////////////////////////////////
// REFERENCE FEATURE ID MODEL
//////////////////////////////////////////////////////////////////////


refID_pModel::refID_pModel(){

	this->loadFiles();
}

refID_pModel* refID_pModel::get_instance(){

	if (refID_instance_ptr == null) {
		refID_instance_ptr = new refID_pModel();
	}
	return refID_instance_ptr;

}

double refID_pModel::getP_refID(int id1){
	return this->p_refID[id1];
}

void refID_pModel::loadFiles(){

	cout << "LOADING THE FILES!" << endl;
	// load the probability model
	ifstream file;


	file.open("thirdparty/coder/brisk/inter/p_refID.bin", ios::in | ios::binary);
	file.read((char*) &p_refID, sizeof(double)*REFID_SIZE);
	file.close();

	cout << "REFERENCE FEATURE ID FILE LOADED!" << endl;
}

refID_pModel* refID_pModel::refID_instance_ptr = null;
