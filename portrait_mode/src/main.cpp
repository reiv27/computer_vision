#include <vector>

#include "utils.hpp"


int main(int argc, char const* argv[]) {
    cv::Mat src = cv::imread(argv[1]);
    cv::Mat bad_mask = cv::imread(argv[2]);

    if (src.empty()) {
        std::cerr << "ERROR! Unable to open image" << std::endl;
        return -1;
    }

    // Create mask with w
    u_int16_t Ny = src.rows;
    u_int16_t Nx = src.cols;
    cv::Mat mask(Ny, Nx, CV_8U, cv::Scalar(0));
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            if (bad_mask.at<cv::Vec3b>(y, x) != cv::Vec3b(0, 0, 0)) {
                mask.at<uchar>(y, x) = 255;
            }
        }
    }

    cv::Mat f = src;
    std::vector<cv::Mat> channels;
    cv::split(f, channels);
    cv::Mat f1 = channels[0];
    cv::Mat f2 = channels[1];
    cv::Mat f3 = channels[2];

    float object_scale = 150;
    float background_scale = 1;
    cv::Mat m(Ny, Nx, CV_8U);
    cv::cvtColor(src, m, cv::COLOR_BGR2GRAY);
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            if (mask.at<uchar>(y, x) != 0) {
                m.at<uchar>(y, x) = object_scale;
            }
            else {
                m.at<uchar>(y, x) = background_scale;
            }
        }
    }
    cv::Mat m_clone = m.clone();
    // m.convertTo(m, CV_32F);

    float l = 0.5;
    cv::Mat ch1 = enhance(f1, m, l);
    cv::Mat ch2 = enhance(f2, m, l);
    cv::Mat ch3 = enhance(f3, m, l);
    std::vector<cv::Mat> rgb_vec = {ch1, ch2, ch3};
    cv::Mat result;
    cv::merge(rgb_vec, result);

    cv::imshow("Original Image, f", f);
    cv::imshow("m", m_clone);
    cv::imshow("Result", result);
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}