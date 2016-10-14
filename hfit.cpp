#include "hfit.h"
#include <math.h>
#include <malloc.h>
#include "opencv2/features2d/features2d.hpp"


extern "C" int mylmdif_(int (*fcn)(int *, int *, double *, double *, int *), int *m, int *n, double *x, double *fvec, double *ftol, double *xtol, double *gtol, int *maxfev, 
	double *epsfcn, double *diag, int *mode, double *factor, int *nprint, int *info, int *nfev, double *fjac, int *ldfjac, int *ipvt, 
	double *qtf, double *wa1, double *wa2, double *wa3, double *wa4);
//
//
///*****************************************************************************
//*****************************************************************************/

std::vector<std::vector<cv::KeyPoint>> key_points_1_g;
std::vector<std::vector<cv::KeyPoint>> key_points_2_g;
std::vector<std::vector<cv::DMatch>> matches_g;
int homography_size;

static double calc_error(int match_i, cv::DMatch match, const cv::Mat& homography) {
		cv::KeyPoint img_pt1 = key_points_1_g[match_i][match.queryIdx];
		cv::KeyPoint img_pt2 = key_points_2_g[match_i][match.trainIdx];

		cv::Vec3d img_pt1_vec3d(img_pt1.pt.x, img_pt1.pt.y, 1.0);

		cv::Vec2d img_pt2_vec2d(img_pt2.pt.x, img_pt2.pt.y);

		cv::Mat calc_pt = homography * cv::Mat(img_pt1_vec3d);
		cv::Vec3d calc_pt_vec3d(calc_pt);

		cv::Vec2d calc_pt_vec2d(calc_pt_vec3d[0] / calc_pt_vec3d[2], calc_pt_vec3d[1] / calc_pt_vec3d[2]);
		double error = cv::norm(calc_pt_vec2d - img_pt2_vec2d);
		return error;
}

static int
lmdifError_(int *m_ptr, int *n_ptr, double *params, double *error, int *)
{
	int nparms = *n_ptr;
	int nerrors = *m_ptr;

	std::vector<cv::Mat> homographies(homography_size);
	

	for (int k = 0; k < homographies.size(); ++k) {
		homographies[k] = cv::Mat(3, 3, CV_64F);
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				 homographies[k].at<double>(i, j) = params[k * 9 + i * 3 + j];
			}
		}
	}

	// H_{01}, H_{12}, H_{23}, H_{34}

	// calc error

	// every match has from 0->1, 0->2, 0->3, 0->4
	// then from 1->2, 1->3, 1->4
	// then from 2->3, 2->4
	// then from 3->4
	// assume everything has 10 points or less

	// cover the 9 cases
	int k = 0;
	// 0->1
	int index = 0;
	for (int i = 0; i < matches_g[0].size(); ++i) {
		auto match = matches_g[0][i];
		cv::Mat homography = homographies[0];
		error[index + i] = calc_error(0, match, homography);
	}
	index += matches_g[0].size();

	// 0->2
	for (int i = 0; i < matches_g[1].size(); ++i) {
		auto match = matches_g[1][i];
		cv::Mat homography = homographies[1] * homographies[0];
		error[index + i] = calc_error(1, match, homography);
	}
	index += matches_g[1].size();

	// 0->3
	for (int i = 0; i < matches_g[2].size(); ++i) {
		auto match = matches_g[2][i];
		cv::Mat homography = homographies[2] * homographies[1] * homographies[0];
		error[index + i] = calc_error(2, match, homography);
	}
	index += matches_g[2].size();

	// 0->4
	for (int i = 0; i < matches_g[3].size(); ++i) {
		auto match = matches_g[3][i];
		cv::Mat homography = homographies[3] * homographies[2] * homographies[1] * homographies[0];
		error[index + i] = calc_error(3, match, homography);
	}
	index += matches_g[3].size();

	// 1->2
	k = 4;
	for (int i = 0; i < matches_g[k].size(); ++i) {
		auto match = matches_g[k][i];
		cv::Mat homography =  homographies[1];
		error[index + i] = calc_error(k, match, homography);
	}
	index += matches_g[k].size();

	// 1->3
	k = 5;
	for (int i = 0; i < matches_g[k].size(); ++i) {
		auto match = matches_g[k][i];
		cv::Mat homography = homographies[2] * homographies[1];
		error[index + i] = calc_error(k, match, homography);
	}
	index += matches_g[k].size();

	// 1->4
	k = 6;
	for (int i = 0; i < matches_g[k].size(); ++i) {
		auto match = matches_g[k][i];
		cv::Mat homography = homographies[3] * homographies[2] * homographies[1];
		error[index + i] = calc_error(k, match, homography);
	}
	index += matches_g[k].size();


	// H_{01}, H_{12}, H_{23}, H_{34}
	// 2->3
	k = 7;
	for (int i = 0; i < matches_g[k].size(); ++i) {
		auto match = matches_g[k][i];
		cv::Mat homography = homographies[2];
		error[index + i] = calc_error(k, match, homography);
	}
	index += matches_g[k].size();

	// 2->4
	k = 8;
	for (int i = 0; i < matches_g[k].size(); ++i) {
		auto match = matches_g[k][i];
		cv::Mat homography = homographies[3] * homographies[2];
		error[index + i] = calc_error(k, match, homography);
	}
	index += matches_g[k].size();

	// 3->4
	k = 9;
	for (int i = 0; i < matches_g[k].size(); ++i) {
		auto match = matches_g[k][i];
		cv::Mat homography = homographies[3];
		error[index + i] = calc_error(k, match, homography);
	}
	index += matches_g[k].size();

	return 1;
}


/*****************************************************************************
*****************************************************************************/
/* Parameters controlling MINPACK's lmdif() optimization routine. */
/* See the file lmdif.f for definitions of each parameter.        */
#define REL_SENSOR_TOLERANCE_ftol    1.0E-6      /* [pix] */
#define REL_PARAM_TOLERANCE_xtol     1.0E-7
#define ORTHO_TOLERANCE_gtol         0.0
#define MAXFEV                       (1000*n)
#define EPSFCN                       1.0E-10 /* was E-16 Do not set to 0! */
#define MODE                         2       /* variables scaled internally */
#define FACTOR                       100.0 


int fit_homographies(std::vector<cv::Mat>& homographies, std::vector<std::vector<cv::KeyPoint>>& key_points_1, std::vector<std::vector<cv::KeyPoint>>& key_points_2,
	std::vector<std::vector<cv::DMatch>>& matches)
{
    /* Parameters needed by MINPACK's lmdif() */
	int     n = 9 * homographies.size();

	int no_of_errors = 0;
	for (int i = 0; i < matches.size(); ++i) {
		no_of_errors += matches[i].size();
	}

	int     m = no_of_errors;
    double *x;
    double *fvec;
    double  ftol = REL_SENSOR_TOLERANCE_ftol;
    double  xtol = REL_PARAM_TOLERANCE_xtol;
    double  gtol = ORTHO_TOLERANCE_gtol;
    int     maxfev = MAXFEV;
    double  epsfcn = EPSFCN;
    double *diag;
    int     mode = MODE;
    double  factor = FACTOR;
    int     ldfjac = m;
    int     nprint = 0;
    int     info;
    int     nfev;
    double *fjac;
    int    *ipvt;
    double *qtf;
    double *wa1;
    double *wa2;
    double *wa3;
    double *wa4;


	 /* copy to globals */
	 key_points_1_g = key_points_1;
	 key_points_2_g = key_points_2;
	 matches_g = matches;
	 homography_size = homographies.size();

    /* allocate stuff dependent on n */
    x    = (double *)calloc(n, sizeof(double));
    diag = (double *)calloc(n, sizeof(double));
    qtf  = (double *)calloc(n, sizeof(double));
    wa1  = (double *)calloc(n, sizeof(double));
    wa2  = (double *)calloc(n, sizeof(double));
    wa3  = (double *)calloc(n, sizeof(double));
    ipvt = (int    *)calloc(n, sizeof(int));

    /* allocate some workspace */
    if (( fvec = (double *) calloc ((unsigned int) m, 
                                    (unsigned int) sizeof(double))) == NULL ) {
       fprintf(stderr,"calloc: Cannot allocate workspace fvec\n");
       exit(-1);
    }

    if (( fjac = (double *) calloc ((unsigned int) m*n,
                                    (unsigned int) sizeof(double))) == NULL ) {
       fprintf(stderr,"calloc: Cannot allocate workspace fjac\n");
       exit(-1);
    }

    if (( wa4 = (double *) calloc ((unsigned int) m, 
                                   (unsigned int) sizeof(double))) == NULL ) {
       fprintf(stderr,"calloc: Cannot allocate workspace wa4\n");
       exit(-1);
    }


    /* copy parameters in as initial values */
	for (int k = 0; k < homographies.size(); ++k) {
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				x[k * 9 + i * 3 + j] = homographies[k].at<double>(i, j);
			}
		}
	}

    /* define optional scale factors for the parameters */
    if ( mode == 2 ) {
		int offset = 0;
		for (int offset = 0; offset<n; offset++) {
			diag[offset] = 1.0;
		}
    }

    /* perform the optimization */ 
    //printf("Starting optimization step...\n");
    mylmdif_ (lmdifError_,
            &m, &n, x, fvec, &ftol, &xtol, &gtol, &maxfev, &epsfcn,
            diag, &mode, &factor, &nprint, &info, &nfev, fjac, &ldfjac,
            ipvt, qtf, wa1, wa2, wa3, wa4);
    double totalerror = 0;
    for (int i=0; i<m; i++) {
       totalerror += fvec[i];
	}
    //printf("\tnum function calls = %i\n", nfev);
    //printf("\tremaining total error value = %f\n", totalerror);
    //printf("\tor %1.2f per point\n", std::sqrt(totalerror) / m);
    //printf("...ended optimization step.\n");

    /* copy result back to parameters array */
	for (int k = 0; k < homographies.size(); ++k) {
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				 homographies[k].at<double>(i, j) = x[k * 9 + i * 3 + j];
			}
		}
	}



    /* release allocated workspace */
    free (fvec);
    free (fjac);
    free (wa4);
    free (ipvt);
    free (wa1);
    free (wa2);
    free (wa3);
    free (qtf);
    free (diag);
    free (x);

	 return (1);
}

