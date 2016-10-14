#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"


extern
int fit_homographies(std::vector<cv::Mat>& homographies, std::vector<std::vector<cv::KeyPoint>>& key_points_1, std::vector<std::vector<cv::KeyPoint>>& key_points_2,
	std::vector<std::vector<cv::DMatch>>& matches);

