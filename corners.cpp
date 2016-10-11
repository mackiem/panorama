#include "Corners.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include "opencv2/highgui/highgui.hpp"


std::vector<cv::Mat> CornerDetector::create_haar_kernel(float sigma) {
	float four_sigma = sigma * 4.f;
	// smallest even integer greater than 4 * sigma
	int integer_greater_than_four_sigma = std::ceil(four_sigma);
	int kernel_size = (integer_greater_than_four_sigma % 2 == 1) ? integer_greater_than_four_sigma + 1 : integer_greater_than_four_sigma;

	cv::Mat horizontal_haar_kernel = cv::Mat::ones(kernel_size, kernel_size, CV_32F);

	for (int row = 0; row < (kernel_size); ++row) {
		for (int col = 0; col < (kernel_size / 2); ++col) {
			horizontal_haar_kernel.at<float>(row, col) = -1.f;
		}
	}

	cv::Mat vertical_haar_kernel = cv::Mat::ones(kernel_size, kernel_size, CV_32F);

	for (int row = kernel_size / 2; row < (kernel_size); ++row) {
		for (int col = 0; col < (kernel_size); ++col) {
			vertical_haar_kernel.at<float>(row, col) = -1.f;
		}
	}

	std::vector<cv::Mat> haar_kernels;
	haar_kernels.push_back(horizontal_haar_kernel);
	haar_kernels.push_back(vertical_haar_kernel);

	std::cout << horizontal_haar_kernel << "\n";
	std::cout << vertical_haar_kernel<< "\n";
	return haar_kernels;
}

std::vector<cv::Vec2i> CornerDetector::calculate_corners(cv::Mat& haar_kernel_x, cv::Mat& haar_kernel_y, cv::Mat& input_img, float sigma, float corner_threshold, cv::Mat& output_img) {

	std::vector<cv::Vec2i> corners;

	cv::Mat dx_img;
	cv::Mat dy_img;

	cv::Mat gray_img;
	cv::cvtColor(input_img, gray_img, CV_RGB2GRAY);

	cv::Mat img;
	cv::normalize(gray_img, img, 0, 1, cv::NORM_MINMAX, CV_64F);


	// create 5 * sigma /2.f borders
	int border = std::ceil(sigma * 5.f / 2.f) ;
	border = (border % 2 == 1) ? border : border + 1;

	cv::GaussianBlur(img, img, cv::Size(5, 5), 0.f, 0.f);

	//cv::Mat dx_img_with_border;
	//cv::Mat dy_img_with_border;
	//cv::copyMakeBorder(dx_img, dx_img_with_border, border, border, border, border, cv::BORDER_REPLICATE);
	//cv::copyMakeBorder(dy_img, dy_img_with_border, border, border, border, border, cv::BORDER_REPLICATE);

	//cv::Mat input_img_with_border;
	//cv::copyMakeBorder(img, input_img_with_border, border, border, border, border, cv::BORDER_REPLICATE);

	cv::filter2D(img, dx_img, CV_64F, haar_kernel_x);
	cv::filter2D(img, dy_img, CV_64F, haar_kernel_y);

	cv::Mat dx_img_8u;
	cv::Mat dy_img_8u;
	cv::convertScaleAbs(dx_img, dx_img_8u);
	cv::convertScaleAbs(dy_img, dy_img_8u);
	cv::imshow("dx_img",dx_img_8u);
	cv::imshow("dy_img",dy_img_8u);
	cv::imwrite("dx_img.jpg",dx_img_8u * 255) ;
	cv::imwrite("dy_img.jpg",dy_img_8u * 255);


	// create C matrix
	output_img = cv::Mat::zeros(img.rows, img.cols, CV_64F);

	for (int row = border; row < input_img.rows - border; ++row) {
		for (int col = border; col < input_img.cols - border; ++col) {
			// construct 5 * sigma window
			double dx_sqr = 0;
			double dy_sqr = 0;
			double dx_times_dy = 0;

			for (int row_5_sigma = -border; row_5_sigma <= border; ++row_5_sigma) {
				for (int col_5_sigma = -border; col_5_sigma <= border; ++col_5_sigma) {
					//int padded_row = row + row_5_sigma + border;
					//int padded_col = col + col_5_sigma + border;

					//double dx = dx_img_with_border.at<double>(padded_row, padded_col);
					//double dy = dy_img_with_border.at<double>(padded_row, padded_col);
					int adj_row = row + row_5_sigma;
					int adj_col = col + col_5_sigma;

					double dx = dx_img.at<double>(adj_row, adj_col);
					double dy = dy_img.at<double>(adj_row, adj_col);
					dx_sqr += std::pow(dx, 2);
					dy_sqr += std::pow(dy, 2);
					dx_times_dy += dx * dy;
				}
			}

			// evaluate C matrix
			double det_C = dx_sqr * dy_sqr - std::pow(dx_times_dy, 2);
			if (det_C > 1e3) {
				// rank is 2, let's evaluate
				double trace_C_sqr = std::pow(dx_sqr + dy_sqr, 2);
				double corner_ratio = det_C / trace_C_sqr;
				//if (corner_ratio > corner_threshold) {
				if (corner_ratio > 0.2) {
					// this is a corner
					output_img.at<double>(row, col) = corner_ratio;
					cv::Vec2i corner(row, col);
					corners.push_back(corner);
				}
			}
		}
	}

	return corners;
}

std::vector<cv::Vec2i> CornerDetector::detect_corners(cv::Mat& img, float sigma, float corner_threshold) {
	auto haar_kernels = create_haar_kernel(sigma);
	cv::Mat corner_img;
	auto corners = calculate_corners(haar_kernels[0], haar_kernels[1], img, sigma, corner_threshold, corner_img);
	auto supressed_corners = non_maximal_supression(corner_img, 10 * sigma, corners);
	return supressed_corners;
	return corners;
}

void CornerDetector::draw_corners(cv::Mat& img, std::vector<cv::Vec2i>& corners) {
	for (auto& corner : corners) {
		//img.at<cv::Vec3b>(corner[0], corner[1]) = cv::Vec3b(0, 255, 0);
		cv::circle(img, cv::Point(corner[1], corner[0]), 3, cv::Scalar(0.0, 255.0, 0.0));
	}
}

std::vector<cv::Vec2i> CornerDetector::non_maximal_supression(cv::Mat& corner_img, int block_size, std::vector<cv::Vec2i> corners) {

	int border = std::ceil(block_size / 2.f) ;
	border = (border % 2 == 1) ? border : border + 1;

	//cv::Mat corner_img_with_border;
	//cv::copyMakeBorder(corner_img, corner_img_with_border, border, border, border, border, cv::BORDER_REPLICATE);

	//for (int row = 0; row < corner_img.rows; ++row) {
	//	for (int col = 0; col < corner_img.cols; ++col) {

	std::vector<cv::Vec2i> supressed_corners;

	for (auto& corner : corners) {

		int row = corner[0];
		int col = corner[1];

		double curr_corner_ratio = corner_img.at<double>(row, col);

		bool is_maximum = true;
		for (int row_5_sigma = -border; row_5_sigma <= border; ++row_5_sigma) {
			for (int col_5_sigma = -border; col_5_sigma <= border; ++col_5_sigma) {
				if (!(row_5_sigma == 0 && col_5_sigma == 0)) {

					//int padded_row = row + row_5_sigma + border;
					//int padded_col = col + col_5_sigma + border;
					int padded_row = row + row_5_sigma;
					int padded_col = col + col_5_sigma;

					if (padded_col >= 0 && padded_col < corner_img.cols
						&& padded_row >= 0 && padded_row < corner_img.rows) {

							double corner_ratio = corner_img.at<double>(padded_row, padded_col);
							//if (row == 7 && col == 53) {
							//	std::cout << "padd row, padd col, corner val : " << padded_row - border << ", " << padded_col - border << ", " << corner_ratio << "\n";
							//}
							if (corner_ratio > curr_corner_ratio) {
								is_maximum = false;
								break;
							}
					}
				}
			}
			if (!is_maximum) {
				break;
			}
		}
		if (is_maximum) {
			//std::cout << "row, col, corner val : " << row << ", " << col << ", " << curr_corner_ratio << "\n";
			supressed_corners.push_back(corner);
		}
	}

	return supressed_corners;
	
}

CornerDetector::CornerDetector(void)
{
}


CornerDetector::~CornerDetector(void)
{
}
