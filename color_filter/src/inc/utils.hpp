#ifndef MY_PROJECT_UTILS_H
#define MY_PROJECT_UTILS_H


#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>


void createMask(const cv::Mat& bad_mask, cv::Mat& mask);
void createFilter(const cv::Mat& src, const cv::Mat& mask, cv::Vec3f& p0, cv::Vec3f& v, float& t_min, float& t_max, float& R);


#endif // MAX_MIN_FILTER_UTILS_H