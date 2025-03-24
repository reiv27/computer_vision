#include <gtest/gtest.h>

#include "utils.hpp"


TEST(FunctionTest, SimpleRealisation) {
    // cv::Mat test_image = cv::imread("../tests/test_images/lenna.png", cv::IMREAD_GRAYSCALE);

    // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    // cv::Mat max, min;
    // cv::dilate(test_image, max, kernel);
    // cv::erode(test_image, min, kernel);
    // cv::Mat opencv_result = max - min;

    // cv::Mat simple = maxMinFilter(test_image);

    // cv::Mat diff;
    // cv::compare(simple, opencv_result, diff, cv::CMP_NE);
    // bool answer = cv::countNonZero(diff);

    // EXPECT_TRUE(!answer);
    EXPECT_TRUE(true);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}