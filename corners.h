#pragma once
#include "utils.h"

class CornerDetector
{
	std::vector<cv::Mat> create_haar_kernel(float sigma);
	std::vector<cv::Vec2i> calculate_corners(cv::Mat& haar_kernel_x, cv::Mat& haar_kernel_y, cv::Mat& input_img, float sigma, float corner_threshold, cv::Mat& output_img);
	std::vector<cv::Vec2i> non_maximal_supression(cv::Mat& corner_img, int block_size, std::vector<cv::Vec2i> corners);

public:
	std::vector<cv::Vec2i> detect_corners(cv::Mat& img, float sigma, float corner_threshold = 0.1);
	void draw_corners(cv::Mat& img, std::vector<cv::Vec2i>& corners );

	CornerDetector(void);
	~CornerDetector(void);
};

