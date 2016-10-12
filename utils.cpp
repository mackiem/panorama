//#include "Utils.h"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include <iostream>
//
//
//
//
//cv::Vec3d Utils::calculate_line(const cv::Point2d& a, const cv::Point2d& b) {
//	cv::Point3d a_hc(a.x, a.y, 1.0);
//	cv::Point3d b_hc(b.x, b.y, 1.0);
//	auto line = calculate_line(a_hc, b_hc);
//	return line;
//}
//
//cv::Vec3d Utils::calculate_line(const cv::Point3d& a, const cv::Point3d& b) {
//	cv::Vec3d line = a.cross(b);
//	line /= line[2];
//	return line;
//}
//
//cv::Vec3d Utils::calculate_intersection(const cv::Vec3d& a, const cv::Vec3d& b) {
//	cv::Vec3d intersection_pt = a.cross(b);
//	return intersection_pt;
//}
//
//
//void Utils::draw_lines(cv::Mat& img, cv::Point2d pts[], cv::Mat& output_img, std::string filename) {
//	output_img = img.clone();
//	cv::Scalar colors[4]; 
//	colors[0] = cv::Scalar(255, 0, 0);
//	colors[1] = cv::Scalar(0, 255, 0);
//	colors[2] = cv::Scalar(0, 0, 255);
//	colors[3] = cv::Scalar(255, 255, 0);
//	for (int i = 0; i < 4; ++i) {
//		cv::Point2d last_pt = pts[(i + 1) % 4];
//		cv::line(output_img, pts[i], last_pt, colors[i], 3);
//	}
//	cv::imwrite(filename, output_img);
//}
//
//void Utils::calculate_ssd(std::vector<cv::Vec2i>& img_1_matches, std::vector<cv::Vec2i>& img_2_matches, cv::Mat& input_img_1, cv::Mat& input_img_2, int border, float threshold,
//	std::vector<cv::KeyPoint>& key_points_1, std::vector<cv::KeyPoint>& key_points_2, std::vector<cv::DMatch>& matches) {
//
//	border = (border % 2 == 1) ? border : border + 1;
//
//	cv::Mat input_img_1_gray;
//	cv::cvtColor(input_img_1, input_img_1_gray, CV_RGB2GRAY);
//	cv::Mat input_img_2_gray;
//	cv::cvtColor(input_img_2, input_img_2_gray, CV_RGB2GRAY);
//
//	cv::Mat input_img_1_with_border;
//	cv::copyMakeBorder(input_img_1_gray, input_img_1_with_border, border, border, border, border, cv::BORDER_REPLICATE);
//	cv::Mat input_img_2_with_border;
//	cv::copyMakeBorder(input_img_2_gray, input_img_2_with_border, border, border, border, border, cv::BORDER_REPLICATE);
//
//	std::vector<MatchingPt> matching_pts;
//	double max_ssd = -DBL_MAX;
//	double min_ssd = DBL_MAX;
//
//	for (auto& img_1_corner : img_1_matches) {
//
//		int img_1_row = img_1_corner[0];
//		int img_1_col = img_1_corner[1];
//
//		for (auto& img_2_corner : img_2_matches) {
//
//			int img_2_row = img_2_corner[0];
//			int img_2_col = img_2_corner[1];
//
//			double ssd = 0.0;
//
//			for (int img_row_border = -border; img_row_border <= border; ++img_row_border) {
//				for (int img_col_border = -border; img_col_border <= border; ++img_col_border) {
//
//					int img_1_padded_row = img_1_row + img_row_border + border;
//					int img_1_padded_col = img_1_col + img_col_border + border;
//
//					int img_2_padded_row = img_2_row + img_row_border + border;
//					int img_2_padded_col = img_2_col + img_col_border + border;
//
//					unsigned char img_1_gray_val = input_img_1_with_border.at<unsigned char>(img_1_padded_row, img_1_padded_col);
//
//					unsigned char img_2_gray_val = input_img_2_with_border.at<unsigned char>(img_2_padded_row, img_2_padded_col);
//
//					ssd += std::pow(std::abs(img_1_gray_val - img_2_gray_val), 2);
//				}
//			}
//
//			max_ssd = std::max(max_ssd, ssd);
//			min_ssd = std::min(min_ssd, ssd);
//
//			MatchingPt matching_pt;
//			matching_pt.score = ssd;
//			matching_pt.pt_1 = img_1_corner;
//			matching_pt.pt_2 = img_2_corner;
//
//			//if (matching_pts.size() < 1000) {
//				matching_pts.push_back(matching_pt);
//			//}
//		}
//	}
//
//
//	// sort values according to SSD for each point
//	std::sort(matching_pts.begin(), matching_pts.end(), [&] (const MatchingPt& lhs, const MatchingPt& rhs)
//	{
//		if (lhs.pt_1[0] == rhs.pt_1[0]) {
//			if (lhs.pt_1[1] == rhs.pt_1[1]) {
//				return lhs.score < rhs.score;
//			}
//			return lhs.pt_1[1] < rhs.pt_1[1];
//		}
//		return lhs.pt_1[0] < rhs.pt_1[0];
//	});
//
//	std::vector<MatchingPt> filtered_matching_pts;
//
//	double ssd_threshold = 2;
//	cv::Vec2i prev_matching_pt(-1, -1);
//
//
//	std::cout << "max ssd : " << max_ssd << " min ssd: " << min_ssd << " threshold : " << threshold << "\n";
//	//std::vector<cv::KeyPoint> key_points_1;
//	//std::vector<cv::KeyPoint> key_points_2;
//	//key_points_1.push_back(key_1);
//	//key_points_2.push_back(key_2);
//
//	//std::vector<cv::DMatch > matches;
//	int index = 0;
//
//	// keep only points that are T * SSD_(max)
//	for (auto& matching_pt : matching_pts) {
//		if (prev_matching_pt[0] == matching_pt.pt_1[0] && prev_matching_pt[1] == matching_pt.pt_1[1]) {
//		} else {
//			if (matching_pt.score < threshold * (min_ssd) || matching_pt.score < 1e-6) {
//				filtered_matching_pts.push_back(matching_pt);
//
//				cv::KeyPoint key_1;
//				key_1.pt = cv::Point2i(matching_pt.pt_1[1], matching_pt.pt_1[0]);
//				cv::KeyPoint key_2;
//				key_2.pt = cv::Point2i(matching_pt.pt_2[1], matching_pt.pt_2[0]);
//
//				key_points_1.push_back(key_1);
//				key_points_2.push_back(key_2);
//
//				cv::DMatch match;
//				match.queryIdx = index;
//				match.trainIdx = index;
//				matches.push_back(match);
//
//
//				index++;
//			} 
//		}
//		prev_matching_pt = matching_pt.pt_1;
//	}
//}
//
//void Utils::calculate_ncc(std::vector<cv::Vec2i>& img_1_matches, std::vector<cv::Vec2i>& img_2_matches, cv::Mat& input_img_1, cv::Mat& input_img_2, int border,  float threshold,
//	std::vector<cv::KeyPoint>& key_points_1, std::vector<cv::KeyPoint>& key_points_2, std::vector<cv::DMatch>& matches) {
//
//	border = (border % 2 == 1) ? border : border + 1;
//
//	cv::Mat input_img_1_gray;
//	cv::cvtColor(input_img_1, input_img_1_gray, CV_RGB2GRAY);
//	cv::Mat input_img_2_gray;
//	cv::cvtColor(input_img_2, input_img_2_gray, CV_RGB2GRAY);
//
//	cv::Mat input_img_1_gray_64F;
//	input_img_1_gray.convertTo(input_img_1_gray_64F, CV_64F);
//	cv::Mat input_img_2_gray_64F;
//	input_img_2_gray.convertTo(input_img_2_gray_64F, CV_64F);
//
//	cv::Mat input_img_1_with_border;
//	cv::copyMakeBorder(input_img_1_gray_64F, input_img_1_with_border, border, border, border, border, cv::BORDER_REPLICATE);
//	cv::Mat input_img_2_with_border;
//	cv::copyMakeBorder(input_img_2_gray_64F, input_img_2_with_border, border, border, border, border, cv::BORDER_REPLICATE);
//
//	std::vector<MatchingPt> matching_pts;
//	double max_ncc = -DBL_MAX;
//	double min_ncc = DBL_MAX;
//
//	for (auto& img_1_corner : img_1_matches) {
//
//		int img_1_row = img_1_corner[0];
//		int img_1_col = img_1_corner[1];
//
//		cv::Rect img_1_window = cv::Rect(img_1_col -border + border, img_1_row - border + border,  2 * border + 1, 2 * border + 1);
//
//		cv::Mat img_1_sub_img = input_img_1_with_border(img_1_window);
//
//		double img_1_mean = cv::mean(img_1_sub_img)[0];
//
//		cv::Mat intensity_deviation_1 = img_1_sub_img - img_1_mean;
//		cv::Mat intensity_diff_sqr_1;
//		cv::pow(intensity_deviation_1, 2.0, intensity_diff_sqr_1);
//		double sqr_sum_1 = cv::sum(intensity_diff_sqr_1)[0];
//
//		for (auto& img_2_corner : img_2_matches) {
//
//			int img_2_row = img_2_corner[0];
//			int img_2_col = img_2_corner[1];
//
//			double ncc = 0.0;
//
//			cv::Rect img_2_window = cv::Rect(img_2_col -border + border, img_2_row - border + border,  2 * border + 1, 2 * border + 1);
//			cv::Mat img_2_sub_img = input_img_2_with_border(img_2_window);
//
//			double img_2_mean = cv::mean(img_2_sub_img)[0];
//
//			cv::Mat intensity_deviation_2 = img_2_sub_img - img_2_mean;
//			cv::Mat intensity_diff_sqr_2;
//			cv::pow(intensity_deviation_2, 2.0, intensity_diff_sqr_2);
//			double sqr_sum_2 = cv::sum(intensity_diff_sqr_2)[0];
//			//std::cout << intensity_diff_sqr_2 << "\n";
//
//			cv::Mat intensity_diff_multiplication;
//			cv::multiply(intensity_deviation_1, intensity_deviation_2, intensity_diff_multiplication);
//
//			//std::cout << intensity_diff_multiplication << "\n";
//
//			double numerator = cv::sum(intensity_diff_multiplication)[0];
//			double denominator = std::sqrt(sqr_sum_1 * sqr_sum_2);
//
//			if (denominator > 1e-6) {
//				ncc =  numerator / denominator;
//			}
//
//			max_ncc = std::max(max_ncc, ncc);
//			min_ncc = std::min(min_ncc, ncc);
//
//			MatchingPt matching_pt;
//			matching_pt.score = ncc;
//			matching_pt.pt_1 = img_1_corner;
//			matching_pt.pt_2 = img_2_corner;
//
//			//if (matching_pts.size() < 1000) {
//				matching_pts.push_back(matching_pt);
//			//}
//		}
//	}
//
//
//	// sort values according to SSD for each point
//	std::sort(matching_pts.begin(), matching_pts.end(), [&] (const MatchingPt& lhs, const MatchingPt& rhs)
//	{
//		if (lhs.pt_1[0] == rhs.pt_1[0]) {
//			if (lhs.pt_1[1] == rhs.pt_1[1]) {
//				return lhs.score > rhs.score;
//			}
//			return lhs.pt_1[1] < rhs.pt_1[1];
//		}
//		return lhs.pt_1[0] < rhs.pt_1[0];
//	});
//
//	std::vector<MatchingPt> filtered_matching_pts;
//
//	double ssd_threshold = 2;
//	cv::Vec2i prev_matching_pt(-1, -1);
//
//	//std::vector<cv::DMatch > matches;
//	int index = 0;
//
//	// keep only points that are T * SSD_(max)
//	for (auto& matching_pt : matching_pts) {
//		if (prev_matching_pt[0] == matching_pt.pt_1[0] && prev_matching_pt[1] == matching_pt.pt_1[1]) {
//		} else {
//			if (matching_pt.score > threshold * (max_ncc) || matching_pt.score < 1e-6) {
//				filtered_matching_pts.push_back(matching_pt);
//
//				cv::KeyPoint key_1;
//				key_1.pt = cv::Point2i(matching_pt.pt_1[1], matching_pt.pt_1[0]);
//				cv::KeyPoint key_2;
//				key_2.pt = cv::Point2i(matching_pt.pt_2[1], matching_pt.pt_2[0]);
//
//				key_points_1.push_back(key_1);
//				key_points_2.push_back(key_2);
//
//				cv::DMatch match;
//				match.queryIdx = index;
//				match.trainIdx = index;
//				matches.push_back(match);
//
//				index++;
//			} 
//		}
//		prev_matching_pt = matching_pt.pt_1;
//	}
//}
//
//void Utils::calculate_surf_ssd(const std::vector<cv::KeyPoint>& keypoints_1, const cv::Mat& descriptor_1, const std::vector<cv::KeyPoint>& keypoints_2, const cv::Mat& descriptor_2, 
//	const cv::Mat& input_img_1, const cv::Mat& input_img_2, float threshold, std::vector<cv::DMatch>& matches) {
//
//
//	std::vector<MatchingSurfPt> matching_pts;
//	double max_ssd = -DBL_MAX;
//	double min_ssd = DBL_MAX;
//
//	//std::cout << descriptor_1.row(0) << " " << descriptor_1.rows << " " << descriptor_1.cols << "\n";
//
//	for (auto i1 = 0; i1 < keypoints_1.size(); ++i1) {
//
//		int img_1_row = keypoints_1[i1].pt.y;
//		int img_1_col = keypoints_1[i1].pt.x;
//
//		cv::Mat keypt_1_descriptor = descriptor_1.row(i1);
//
//		for (auto i2 = 0; i2 < keypoints_2.size(); ++i2) {
//
//			int img_2_row = keypoints_2[i2].pt.y;
//			int img_2_col = keypoints_2[i2].pt.x;
//
//			double ssd = 0.0;
//
//			cv::Mat keypt_2_descriptor = descriptor_2.row(i2);
//
//			for (int k = 0; k < keypt_1_descriptor.cols; ++k) {
//				auto des_1 = keypt_1_descriptor.at<float>(0, k);
//				auto des_2 = keypt_2_descriptor.at<float>(0, k);
//				ssd += std::pow(std::abs(des_1 - des_2), 2);
//			}
//
//			max_ssd = std::max(max_ssd, ssd);
//			min_ssd = std::min(min_ssd, ssd);
//
//			MatchingSurfPt matching_pt;
//			matching_pt.score = ssd;
//			matching_pt.index_1 = i1;
//			matching_pt.index_2 = i2;
//
//			//if (matching_pts.size() < 1000) {
//				matching_pts.push_back(matching_pt);
//			//}
//		}
//	}
//
//
//	// sort values according to SSD for each point
//	std::sort(matching_pts.begin(), matching_pts.end(), [&] (const MatchingSurfPt& lhs, const MatchingSurfPt& rhs)
//	{
//		if (lhs.index_1 == rhs.index_1) {
//			return lhs.score < rhs.score;
//		}
//		return lhs.index_1 < rhs.index_1;
//	});
//
//	std::vector<MatchingSurfPt> filtered_matching_pts;
//
//	double ssd_threshold = 2;
//	//cv::Vec2i prev_matching_pt(-1, -1);
//	int prev_index = -1;
//
//
//	std::cout << "max ssd : " << max_ssd << " min ssd: " << min_ssd << " threshold : " << threshold << "\n";
//	//std::vector<cv::KeyPoint> key_points_1;
//	//std::vector<cv::KeyPoint> key_points_2;
//	//key_points_1.push_back(key_1);
//	//key_points_2.push_back(key_2);
//
//	//std::vector<cv::DMatch > matches;
//
//	// keep only points that are T * SSD_(max)
//	for (auto i = 0; i < matching_pts.size(); ++i) {
//		auto matching_pt = matching_pts[i];
//		if (prev_index == matching_pt.index_1) {
//		} else {
//			if (matching_pt.score < threshold * (min_ssd)) {
//				cv::DMatch match;
//				match.queryIdx = matching_pt.index_1;
//				match.trainIdx = matching_pt.index_2;
//				matches.push_back(match);
//			} 
//		}
//		prev_index = matching_pt.index_1;
//	}
//}
//
//void Utils::calculate_surf_ncc(const std::vector<cv::KeyPoint>& keypoints_1, const cv::Mat& descriptor_1, const std::vector<cv::KeyPoint>& keypoints_2, const cv::Mat& descriptor_2, 
//	const cv::Mat& input_img_1, const cv::Mat& input_img_2, float threshold, std::vector<cv::DMatch>& matches) {
//
//
//	std::vector<MatchingSurfPt> matching_pts;
//	double max_ncc = -DBL_MAX;
//	double min_ncc = DBL_MAX;
//
//	//std::cout << descriptor_1.row(0) << " " << descriptor_1.rows << " " << descriptor_1.cols << "\n";
//
//	for (auto i1 = 0; i1 < keypoints_1.size(); ++i1) {
//
//		int img_1_row = keypoints_1[i1].pt.y;
//		int img_1_col = keypoints_1[i1].pt.x;
//
//		cv::Mat keypt_1_descriptor = descriptor_1.row(i1);
//
//		for (auto i2 = 0; i2 < keypoints_2.size(); ++i2) {
//
//			int img_2_row = keypoints_2[i2].pt.y;
//			int img_2_col = keypoints_2[i2].pt.x;
//
//			double ncc = 0.0;
//
//			cv::Mat keypt_2_descriptor = descriptor_2.row(i2);
//
//			double avg_1 = 0.0;
//			double avg_2 = 0.0;
//
//			for (int k = 0; k < keypt_1_descriptor.cols; ++k) {
//				avg_1 += keypt_1_descriptor.at<float>(0, k);
//				avg_2 += keypt_2_descriptor.at<float>(0, k);
//			}
//			avg_1 /= keypt_1_descriptor.cols;
//			avg_2 /= keypt_2_descriptor.cols;
//
//			cv::Mat variance_mat_1(1, keypt_1_descriptor.cols, CV_32F);
//			cv::Mat variance_mat_2(1, keypt_2_descriptor.cols, CV_32F);
//
//			for (int k = 0; k < keypt_1_descriptor.cols; ++k) {
//				variance_mat_1.at<float>(0, k) = keypt_1_descriptor.at<float>(0, k) - avg_1;
//				variance_mat_2.at<float>(0, k) = keypt_2_descriptor.at<float>(0, k) - avg_2;
//			}
//
//			cv::Mat variance_mat_1_sqr;
//			cv::Mat variance_mat_2_sqr;
//
//			cv::pow(variance_mat_1, 2.0, variance_mat_1_sqr);
//			cv::pow(variance_mat_2, 2.0, variance_mat_2_sqr);
//
//			double var_1_sqr_sum = cv::sum(variance_mat_1_sqr)[0];
//			double var_2_sqr_sum = cv::sum(variance_mat_2_sqr)[0];
//
//			cv::Mat var_1_times_var_2;
//			cv::multiply(variance_mat_1, variance_mat_2, var_1_times_var_2);
//			double numerator = cv::sum(var_1_times_var_2)[0];
//			double denominator = std::sqrt(var_1_sqr_sum * var_2_sqr_sum);
//
//			if (denominator > 1e-8) {
//				ncc = numerator / denominator;
//			}
//
//
//			max_ncc = std::max(max_ncc, ncc);
//			min_ncc = std::min(min_ncc, ncc);
//
//			MatchingSurfPt matching_pt;
//			matching_pt.score = ncc;
//			matching_pt.index_1 = i1;
//			matching_pt.index_2 = i2;
//
//			//if (matching_pts.size() < 1000) {
//				matching_pts.push_back(matching_pt);
//			//}
//		}
//	}
//
//
//	// sort values according to SSD for each point
//	std::sort(matching_pts.begin(), matching_pts.end(), [&] (const MatchingSurfPt& lhs, const MatchingSurfPt& rhs)
//	{
//		if (lhs.index_1 == rhs.index_1) {
//			return lhs.score > rhs.score;
//		}
//		return lhs.index_1 < rhs.index_1;
//	});
//
//	std::vector<MatchingSurfPt> filtered_matching_pts;
//
//	double ssd_threshold = 2;
//	//cv::Vec2i prev_matching_pt(-1, -1);
//	int prev_index = -1;
//
//
//	std::cout << "max ncc : " << max_ncc << " min ncc: " << min_ncc << " threshold : " << threshold << "\n";
//	//std::vector<cv::KeyPoint> key_points_1;
//	//std::vector<cv::KeyPoint> key_points_2;
//	//key_points_1.push_back(key_1);
//	//key_points_2.push_back(key_2);
//
//	//std::vector<cv::DMatch > matches;
//
//	// keep only points that are T * SSD_(max)
//	for (auto i = 0; i < matching_pts.size(); ++i) {
//		auto matching_pt = matching_pts[i];
//		if (prev_index == matching_pt.index_1) {
//		} else {
//			if (matching_pt.score > threshold * (max_ncc)) {
//				cv::DMatch match;
//				match.queryIdx = matching_pt.index_1;
//				match.trainIdx = matching_pt.index_2;
//				matches.push_back(match);
//			} 
//		}
//		prev_index = matching_pt.index_1;
//	}
//}
//
//
