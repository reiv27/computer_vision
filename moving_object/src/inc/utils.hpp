#ifndef MY_PROJECT_UTILS_HPP
#define MY_PROJECT_UTILS_HPP


#include <iostream>
#include <vector>
#include <random> 
#include <opencv2/opencv.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/core/hal/intrin_sse.hpp>


void makeGaussianPyramide(cv::Mat const& im, std::vector<cv::Mat>& pyr, int nlevels = -1);
void makeLaplacianPyramide(cv::Mat const& im, std::vector<cv::Mat>& pyr, int nlevels);
void pyramdalMergePair(const cv::Mat& im1, const cv::Mat& im2, const cv::Mat& mask, cv::Mat& result, int nlevels=-1);
void alphaBlend(const cv::Mat& u1, const cv::Mat& u2, const cv::Mat& mask, cv::Mat& result);

void findCornersOnSecondImage(const cv::Mat& im_1, const cv::Mat& im_2, const std::vector<cv::Point>& corners_1, std::vector<cv::Point2f>& corners_2, std::vector<bool>& mask);

void RANSAC(const std::vector<cv::Point>& corners_1, const std::vector<cv::Point2f>& corners_2, const std::vector<bool>& mask, std::vector<bool>& result_mask, float epsilon=100, float part_of_data=0.9);

cv::Mat finding_H(const std::vector<cv::Point>& corners_1, const std::vector<cv::Point2f>& corners_2, const std::vector<bool>& mask);

void appendImage(const cv::Mat& src, cv::Mat& result, const cv::Mat& H, int overlap=25, bool border=false);


#endif // MAX_MIN_FILTER_UTILS_HPP