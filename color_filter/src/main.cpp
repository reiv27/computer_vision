#include <chrono>
#include <vector>

#include "inc/utils.hpp"


int main(int argc, char const* argv[]) {

    if (argc < 4) {
        std::cerr << "ERROR! Wrong usage!" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <path_to_train_image> <path_to_mask_image> <path_to_test_image>" << std::endl;
        return -1;
    }

    cv::Mat src = cv::imread(argv[1]);  // Train image
    cv::Mat bad_mask = cv::imread(argv[2]);  // Bad mask (not binary)
    cv::Mat test_image = cv::imread(argv[3]);  // Test image

    if (src.empty()) {
        std::cerr << "ERROR! Unable to open images" << std::endl;
        return -1;
    }

    int Ny = src.rows;
    int Nx = src.cols;
    cv::Mat mask(Ny, Nx, CV_8U, cv::Scalar(0));

    createMask(bad_mask, mask);

    cv::Vec3f p0, v;
    float t_min, t_max, R;
    createFilter(src, mask, p0, v, t_min, t_max, R);

    // Testing
    float v_norm = cv::norm(v);
    float v_norm_2 = v_norm * v_norm;
    cv::Mat filtered_image(Ny, Nx, CV_8U, cv::Scalar(0));
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            cv::Vec3f p = test_image.at<cv::Vec3b>(y, x);            
            float t = ((p - p0).t() * v)[0] / v_norm_2;
            float r = cv::norm((p - p0).cross(v)) / v_norm_2;
            if ((t <= t_max) && (t >= t_min) && (r <= R)) {
                filtered_image.at<uchar>(y, x) = 255;
            }
        }
    }

    cv::imshow("Train Image", src);
    cv::imshow("Mask", mask);
    cv::imshow("Test Image", test_image);
    cv::imshow("Filtered Image", filtered_image);
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}