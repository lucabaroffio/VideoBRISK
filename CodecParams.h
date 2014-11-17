#ifndef _CODECPARAMS_H_
#define _CODECPARAMS_H_

#include <iostream>
#include <fstream>
#include <vector>
#include "global.h"

#define null 0
#define AC_PRECISION 1000
#define BRISK_LENGTH_BITS 128
#define BRISK_LENGTH_BYTES BRISK_LENGTH_BITS/8

using namespace std;


// PROBABILITY MODEL FOR BRISK DESCRIPTOR
// Important: uses a singleton class, in order to load the model only the first time the object is created
// --------------------------------------------------------------------------------------------------------
class BRISK_pModel {
private:

    double P0  [1776];
	double P0c0[1776][1776];
	double P0c1[1776][1776];
	int    idxs[BRISK_LENGTH_BITS];

	static BRISK_pModel* instance_ptr;

	void loadFiles();

public:

	BRISK_pModel();
	~BRISK_pModel() {};
	static BRISK_pModel* get_instance();
	double getP0(int id);
	double getP0c0(int id1, int id2);
	double getP0c1(int id1, int id2);
    int getPairs(vector<int> &pairs);
};


class BRISKinter_pModel {
private:

    double P0  [1776];
	double P0c0[1776][1776];
	double P0c1[1776][1776];
	int    inter_idxs[BRISK_LENGTH_BITS];
	int    relative_idxs[BRISK_LENGTH_BITS];

	static BRISKinter_pModel* inter_instance_ptr;

	void loadFiles();

public:

	BRISKinter_pModel();
	~BRISKinter_pModel() {};
	static BRISKinter_pModel* get_instance();
	double getP0(int id);
	double getP0c0(int id1, int id2);
	double getP0c1(int id1, int id2);
    int getPairs(vector<int> &pairs);
    int get_inter_idx(int id);
    int getRelativePairs(vector<int> &pairs);
    int get_inter_relative_idx(int id);
};

class motionvectors_pModel {
private:

	static const double PREC_x = 0.25;
	static const double PREC_y = 0.25;
	static const double PREC_s = 0.25;
//	static const int MVX_SIZE = 241;
//	static const int MVY_SIZE = 241;
//	static const int MVS_SIZE = 41;
	double p_mv_x[MVXSIZE];
	double p_mv_y[MVYSIZE];
	double p_mv_s[MVSSIZE];

	static motionvectors_pModel* mv_instance_ptr;

	void loadFiles();

public:

	motionvectors_pModel();
	~motionvectors_pModel() {};
	static motionvectors_pModel* get_instance();

	double getP_mv_x(int id1);
	double getP_mv_y(int id1);
	double getP_mv_s(int id1);

	double getP_mv_x(double val);
    double getP_mv_y(double val);
	double getP_mv_s(double val);

	double* getP_mv_x();
	double* getP_mv_y();
	double* getP_mv_s();



	const double getPrecS() const {
		return PREC_s;
	}

	const double getPrecX() const {
		return PREC_x;
	}

	const double getPrecY() const {
		return PREC_y;
	}
};


class refID_pModel {
private:

	// static const int REFID_SIZE = 301;
	double p_refID[REFID_SIZE];

	static refID_pModel* refID_instance_ptr;

	void loadFiles();

public:

	refID_pModel();
	~refID_pModel() {};
	static refID_pModel* get_instance();

	double getP_refID(int id1);

};

#endif
