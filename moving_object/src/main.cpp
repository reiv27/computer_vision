#include <cmath>

#include "inc/utils.hpp"


int main(int argc, char const* argv[]) {
    cv::Mat im_1 = cv::imread(argv[1]);
    cv::Mat im_2 = cv::imread(argv[2]);

    if (argc < 3) {
        std::cerr << "ERROR! Image is not found!" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <path_to_image>" << std::endl;
        return -1;
    }

    if (im_1.empty() || im_2.empty()) {
        std::cerr << "ERROR! Unable to open image" << std::endl;
        return -1;
    }

    if (im_1.size() != im_2.size()) {
        std::cerr << "ERROR! Wrong image sizes" << std::endl;
        return -1;
    }

    // Converting to gray
    cv::Mat gray_im_1;
    cv::cvtColor(im_1, gray_im_1, cv::COLOR_BGR2GRAY);
    // cv::equalizeHist(gray_im_1, gray_im_1);
    std::vector<cv::Point> corners_1;
    cv::goodFeaturesToTrack(gray_im_1, corners_1, 100, 0.5, 30);

    std::vector<cv::Point2f> corners_2(corners_1.size(), cv::Point2f(0, 0));
    std::vector<bool> mask(corners_1.size(), true);
    findCornersOnSecondImage(im_1, im_2, corners_1, corners_2, mask);

    std::vector<bool> mask_for_MNK(corners_1.size(), false);
    RANSAC(corners_1, corners_2, mask, mask_for_MNK, 10, 0.9);

    cv::Mat H = finding_H(corners_1, corners_2, mask_for_MNK);

    int koef = 1;
    int Nx = im_1.cols * koef;
    int Ny = im_1.rows * koef;
    cv::Mat warp_im2;    
    cv::warpPerspective(im_2, warp_im2, H, cv::Size(Nx,Ny));

    for (int i = 0; i != corners_1.size(); ++i) {
        if (mask[i]) {
            cv::circle(im_1, corners_1[i], 5, cv::Scalar(255, 0, 0), 5);
            cv::circle(im_2, corners_1[i], 5, cv::Scalar(255, 0, 0), 5);
            cv::circle(im_2, corners_2[i], 5, cv::Scalar(0, 255, 0), 5);
            cv::circle(warp_im2, corners_1[i], 5, cv::Scalar(0, 0, 255), 5);
        }
    }

    cv::imshow("Photo 1", im_1);
    cv::imshow("Photo 2", im_2);
    cv::imshow("Warp Im 2", warp_im2);
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}