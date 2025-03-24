#include <vector>
#include <cmath>

#include "utils.hpp"


int main(int argc, char const* argv[]) {
    cv::Mat src = cv::imread(argv[1]);

    if (src.empty()) {
        std::cerr << "ERROR! Unable to open image" << std::endl;
        return -1;
    }

    cv::Size k_size(5, 5);
    // Converting to gray
    u_int16_t Ny = src.rows;
    u_int16_t Nx = src.cols;
    cv::Mat gray_image;
    cv::Mat blur;
    cv::Mat binary;

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> output_hierarchy;
    std::vector<std::vector<cv::Point>> approx_contours = {};
    std::vector<cv::Point> approx;

    cv::cvtColor(src, gray_image, cv::COLOR_BGR2GRAY);

    cv::adaptiveThreshold(gray_image, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 75, 10);

    cv::Mat binary_clone = binary.clone();
    cv::findContours(binary_clone, contours, output_hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);    

    for (size_t i = 0; i != contours.size(); i++ ) {
        cv::approxPolyDP(contours[i], approx, 3, true);
        if (approx.size() == 6)
            approx_contours.push_back(approx);
    }

    // Finding means for every contours
    std::vector<cv::Point> means;
    for (size_t y = 0; y != approx_contours.size(); ++y) {
        cv::Point2i sum(0, 0);
        for (size_t x = 0; x != 6; ++x) {   
            sum.x += approx_contours[y][x].x;
            sum.y += approx_contours[y][x].y;
        }
        means.push_back(sum / 6);
    }

    std::vector<int> mins_k = {};
    for (size_t y = 0; y != approx_contours.size(); ++y) {
        int64_t dx = approx_contours[y][0].x - means[0].x;
        int64_t dy = approx_contours[y][0].y - means[0].y;
        uint64_t min = std::sqrt(std::pow(dx,2) + std::pow(dy,2));
        int k = 0;
        uint64_t norm;
        for (size_t x = 1; x != 6; ++x) {
            dx = approx_contours[y][x].x - means[y].x;
            dy = approx_contours[y][x].y - means[y].y;
            norm = std::sqrt(std::pow(dx,2) + std::pow(dy,2));
            if (min >= norm) {
                min = norm;
                k = x;
            }
        }
        mins_k.push_back(k);
    }

    std::vector<std::vector<cv::Point>> result_contours = {};
    size_t n = approx_contours.size();
    uint8_t epsilon = 10;
    for (size_t y = 0; y != n; ++y) {
        size_t i = mins_k[y];
        size_t index_minus_2 = (i - 2 + 6) % 6;
        size_t index_plus_2 = (i + 2 + 6) % 6;
        cv::Point minus_2 = approx_contours[y][index_minus_2];
        cv::Point plus_2 = approx_contours[y][index_plus_2];
        uint64_t mean_x = (minus_2.x + plus_2.x) / 2;
        uint64_t mean_y = (minus_2.y + plus_2.y) / 2;
        int64_t dx = approx_contours[y][mins_k[y]].x - mean_x;
        int64_t dy = approx_contours[y][mins_k[y]].y - mean_y;
        uint64_t norm = std::sqrt(std::pow(dx,2) + std::pow(dy,2));

        if (norm < epsilon) {
            result_contours.push_back(approx_contours[y]);
        }
    }

    cv::RNG rng(12345);
    cv::Mat drawing = src.clone();
    for (auto i = 0; i < result_contours.size(); i++ ) {
        cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        cv::drawContours(drawing, result_contours, i, color, 3);
    }

    cv::imshow("Original Image", src);
    cv::imshow("Gray Image", gray_image);
    cv::imshow("Binary", binary);
    cv::imshow("Finded Contours", drawing);
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}