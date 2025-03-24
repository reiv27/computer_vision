#include <chrono>

#include "utils.hpp"


int main(int argc, char const* argv[]) {
    if (argc < 2) {
        std::cerr << "ERROR! Image is not found!" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <path_to_image>" << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread(argv[1]);

    if (image.empty()) {
        std::cerr << "ERROR! Unable to open image" << std::endl;
        return -1;
    }

    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    // MaxMinFilter
    auto t1 = std::chrono::system_clock::now();
    cv::Mat simple_filtered = maxMinFilter(gray_image);
    auto t2 = std::chrono::system_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "maxMinFilter: " << dt << " ms\n";

    // Fast MaxMinFilter
    t1 = std::chrono::system_clock::now();
    cv::Mat fast_filtered = fastMaxMinFilter(gray_image);
    t2 = std::chrono::system_clock::now();
    dt = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "fast_maxMinFilter: " << dt << " ms\n";
    
    cv::Mat diff = simple_filtered - fast_filtered;

    // OpenCV functions
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::Mat max, min;

    t1 = std::chrono::system_clock::now();
    cv::dilate(gray_image, max, kernel);
    cv::erode(gray_image, min, kernel);
    cv::Mat opencv_result = max - min;
    t2 = std::chrono::system_clock::now();
    dt = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "OpenCV functions: " << dt << " ms\n";

    cv::Mat diff_cv;
    cv::absdiff(opencv_result, simple_filtered, diff_cv);

    cv::imshow("Image", gray_image);
    cv::imshow("Simple", simple_filtered);
    cv::imshow("Fast", fast_filtered);
    // cv::imshow("Difference", diff);
    // cv::imshow("OpenCV", opencv_result);
    // cv::imshow("OpenCV vs Simple", diff_cv);
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}