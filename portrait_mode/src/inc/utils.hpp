#ifndef MY_PROJECT_UTILS_H
#define MY_PROJECT_UTILS_H


#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/core/hal/intrin_sse.hpp>

cv::Mat enhance(cv::Mat const& f, cv::Mat const& w, float lam);

#endif // MAX_MIN_FILTER_UTILS_H