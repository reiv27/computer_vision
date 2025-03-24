#include "inc/utils.hpp"


void makeGaussianPyramide(cv::Mat const& im, std::vector<cv::Mat>& pyr, int nlevels) {
    pyr.clear();

    pyr.push_back(im);
    nlevels -= 1;
    
    cv::Mat tmp;
    while (nlevels != 0) {
        cv::pyrDown(pyr.back(), tmp);
        pyr.push_back(tmp.clone());
        if (std::min(tmp.size[0], tmp.size[1]) <= 2)
            break;
        nlevels -= 1;
    }
}


void makeLaplacianPyramide(cv::Mat const& im, std::vector<cv::Mat>& pyr, int nlevels) {
    pyr.clear();

    cv::Mat im1;
    im.convertTo(im1, CV_16S);

    pyr.push_back(im);
    nlevels -= 1;
    
    cv::Mat im2, im3;
    cv::Mat layer;
    while (std::min(im1.rows, im1.cols) > 2 && (nlevels != 0)) {
        cv::pyrDown(im1, im2);;
        cv::pyrUp(im2, im3, cv::Size(im1.cols, im1.rows));
        cv::subtract(im1, im3, layer);
        pyr.push_back(layer);
        im1 = im2;
        if (nlevels != 0) {
            nlevels -= 1;
        }
    }
}


void findCornersOnSecondImage(const cv::Mat& im_1, const cv::Mat& im_2, const std::vector<cv::Point>& corners_1, std::vector<cv::Point2f>& corners_2, std::vector<bool>& mask) {
    cv::Mat gray_im_1;
    cv::Mat gray_im_2;
    cv::cvtColor(im_1, gray_im_1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(im_2, gray_im_2, cv::COLOR_BGR2GRAY);

    int n = corners_1.size();
    std::vector<cv::Point2f> displacements(n, cv::Point(0, 0));

    std::vector<cv::Mat> pyr_1;
    std::vector<cv::Mat> pyr_2;
    uint8_t levels = 4;
    makeGaussianPyramide(gray_im_1, pyr_1, levels);
    makeGaussianPyramide(gray_im_2, pyr_2, levels);

    cv::Point p1, p2;
    std::vector<cv::Point> pyr_corners;
    cv::Mat result;
    for (int j = levels; j != 0; --j) {
        pyr_corners.clear();
        corners_2.clear();
        for (int i = 0; i != n; ++i) {

            if (!mask[i]) {
                continue;
            }

            p1.x = corners_1[i].x / std::pow(2, j-1);
            p1.y = corners_1[i].y / std::pow(2, j-1);
            if (j == 5) {
                p2.x = p1.x;
                p2.y = p1.y;
            }
            else {
                p2.x = p1.x + displacements[i].x * 2;
                p2.y = p1.y + displacements[i].y * 2;
            }
            pyr_corners.push_back(p1);

            int window_size = 5;
            int half_window_1 = window_size / 2;
            int window_size_2 = 20;
            int half_window_2 = window_size_2 / 2;
            cv::Rect roi_1(p1.x - half_window_1, p1.y - half_window_1, window_size, window_size);
            cv::Rect roi_2(p2.x - half_window_2, p2.y - half_window_2, window_size_2, window_size_2);

            if (roi_1.x < 0 || roi_1.y < 0 || roi_1.x + roi_1.width > pyr_1[j-1].cols || roi_1.y + roi_1.height > pyr_1[j-1].rows) {
                continue;
                mask[i] = false;
            }

            if (roi_2.x < 0 || roi_2.y < 0 || roi_2.x + roi_2.width > pyr_2[j-1].cols || roi_2.y + roi_2.height > pyr_2[j-1].rows) {
                continue;
                mask[i] = false;
            }

            cv::matchTemplate(pyr_2[j-1](roi_2), pyr_1[j-1](roi_1), result, cv::TM_CCOEFF_NORMED);
            double min_val, max_val;
            cv::Point min_loc, max_loc;
            cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);

            float x2 = p2.x - half_window_2 + max_loc.x;
            float y2 = p2.y - half_window_2 + max_loc.y;
            displacements[i].x = x2 - p1.x;
            displacements[i].y = y2 - p1.y;
        }
    }

    for (int i = 0; i != displacements.size(); ++i) {
        if (mask[i]) {
            float x2 = corners_1[i].x + displacements[i].x;
            float y2 = corners_1[i].y + displacements[i].y;
            corners_2.push_back(cv::Point2f(x2, y2));
        }
    }
}


void RANSAC(const std::vector<cv::Point>& corners_1, const std::vector<cv::Point2f>& corners_2, const std::vector<bool>& mask, std::vector<bool>& result_mask, float epsilon, float part_of_data) {
    std::vector<cv::Point> q;
    std::vector<cv::Point2f> p;
    for (int i = 0; i < corners_1.size(); ++i) {
        if (mask[i]) {
            p.push_back(corners_1[i]);
            q.push_back(corners_2[i]);
        }
    }

    int n = p.size();
    int cols = 9; 
    int A_rows = 8;
    cv::Mat A(A_rows, cols, CV_32F, cv::Scalar(0.0));

    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    int min = 0, max = n-1, tmp = 0;
    std::uniform_int_distribution<> distrib(min, max);
    
    while (true) {
        for (int i = 0; i < n; ++i) {
            result_mask[i] = false;
        }

        for (int i = 0; i < A.rows; i=i+2) {
            int rn = distrib(gen);  // Random number

            A.at<float>(i, 0) = q[rn].x;               //qx
            A.at<float>(i, 1) = q[rn].y;               //qy
            A.at<float>(i, 2) = 1.0;  
            A.at<float>(i, 6) = -q[rn].x * p[rn].x;    // -qx * px
            A.at<float>(i, 7) = -q[rn].y * p[rn].x;    // -qy * px
            A.at<float>(i, 8) = -p[rn].x;              // -px
    
            A.at<float>(i+1, 3) = q[rn].x;             //qx
            A.at<float>(i+1, 4) = q[rn].y;             //qy
            A.at<float>(i+1, 5) = 1.0;
            A.at<float>(i+1, 6) = -q[rn].x * p[rn].y;  // -qx * py
            A.at<float>(i+1, 7) = -q[rn].y * p[rn].y;  // -qy * py
            A.at<float>(i+1, 8) = -p[rn].y;            // -py
        }

        // Finding singularx values and H
        cv::SVD svd(A, cv::SVD::FULL_UV);
        cv::Mat H = svd.vt.row(8);
        H = H.reshape(1, 3);
        H /= H.at<float>(2, 2);

        int count = 0;
        int t = 0;
        cv::Mat p_true(3, 1, CV_32F);
        cv::Mat p_est(3, 1, CV_32F);
        cv::Mat q_true(3, 1, CV_32F);
        double diff;
        for (int i = 0; i < n; ++i) {
            q_true.at<float>(0,0) = q[i].x;
            q_true.at<float>(1,0) = q[i].y;
            q_true.at<float>(2,0) = 1.0;
            p_true.at<float>(0,0) = p[i].x;
            p_true.at<float>(1,0) = p[i].y;
            p_true.at<float>(2,0) = 1.0;

            p_est = H * q_true;
            diff = cv::norm(p_est - p_true);
            if (diff < epsilon) {
                ++count;
                result_mask[i] = true;
            }
            t = count / n;
        }

        if (t > part_of_data)
            break;
    }
}


cv::Mat finding_H(const std::vector<cv::Point>& corners_1, const std::vector<cv::Point2f>& corners_2, const std::vector<bool>& mask) {
    std::vector<cv::Point> q;
    std::vector<cv::Point2f> p;
    for (int i = 0; i < corners_1.size(); ++i) {
        if (mask[i]) {
            p.push_back(corners_1[i]);
            q.push_back(corners_2[i]);
        }
    }

    int n = p.size();
    int cols = 9; 
    int A_rows = 2 * n;
    cv::Mat A(A_rows, cols, CV_32F, cv::Scalar(0.0));
    
    int k = 0;
    for (int i = 0; i < A.rows; i=i+2) {
        A.at<float>(i, 0) = q[k].x;               //qx
        A.at<float>(i, 1) = q[k].y;               //qy
        A.at<float>(i, 2) = 1.0;  
        A.at<float>(i, 6) = -q[k].x * p[k].x;    // -qx * px
        A.at<float>(i, 7) = -q[k].y * p[k].x;    // -qy * px
        A.at<float>(i, 8) = -p[k].x;              // -px

        A.at<float>(i+1, 3) = q[k].x;             //qx
        A.at<float>(i+1, 4) = q[k].y;             //qy
        A.at<float>(i+1, 5) = 1.0;
        A.at<float>(i+1, 6) = -q[k].x * p[k].y;  // -qx * py
        A.at<float>(i+1, 7) = -q[k].y * p[k].y;  // -qy * py
        A.at<float>(i+1, 8) = -p[k].y;            // -py
    
        ++k;
    }

    // Finding singularx values and H
    cv::SVD svd(A, cv::SVD::FULL_UV);
    cv::Mat H = svd.vt.row(8);
    H = H.reshape(1, 3);
    H /= H.at<float>(2, 2);

    return H;
}


// void appendImage(const cv::Mat& src, cv::Mat& result, const cv::Mat& H, int overlap, bool border) {
//     int h = result.rows;
//     int w = result.cols;
//     cv::Mat mask(w, h, CV_32F, cv::Scalar(255));
//     for (int y = 0; y < h; ++y) {
//         for (int x = 0; x < w; ++x) {
//             if (x < overlap or x > w-overlap)
//                 mask.at<char>(y, x) = 0;
//         }
//     }
//     cv::Mat mask_wrapped;
//     cv::warpPerspective(mask, mask_wrapped, H, cv::Size(w, h));
//     cv::GaussianBlur(mask_wrapped, mask_wrapped, cv::Size(2*overlap-1, 2*overlap-1), 0, 0);
//     cv::Mat src_wrapped;
//     cv::warpPerspective(src, src_wrapped, H, cv::Size(w, h));
//     alphaBlend(result, src_wrapped, mask_wrapped, result);
// }


// void pyramdalMergePair(const cv::Mat& im1, const cv::Mat& im2, const cv::Mat& mask, cv::Mat& result, int nlevels=-1) {
//     std::vector<cv::Mat> pyrm, pyr1, pyr2;
//     makeGaussianPyramide(mask, pyrm, nlevels);
//     makeLaplacianPyramide(im1, pyr1, nlevels);
//     makeLaplacianPyramide(im2, pyr2, nlevels);

//     cv::GaussianBlur(mask, mask, cv::Size(7, 7), 0);

//     cv::Mat u1, u2, m;
//     pyr1.back().convertTo(u1, CV_32S);
//     pyr2.back().convertTo(u2, CV_32S);
//     m = 
//     cv::merg
// }


// void alphaBlend(const cv::Mat& u1, const cv::Mat& u2, const cv::Mat& mask, cv::Mat& result) {
//     cv::Mat u1_16u, u2_16u, u, mask_16u;
//     u1.convertTo(u1_16u, CV_16U);
//     u2.convertTo(u2_16u, CV_16U);
//     mask.convertTo(mask_16u, CV_16U);

//     cv::Mat term1 = 255 - mask_16u;
//     cv::multiply(term1, u1_16u, term1);
//     // cv::Mat term1_1 = term1 / 255;
//     // cv::Mat term2 = mask_16u * u2_16u / 255;
//     // u = (term1 + term2 + 127) / 255;

//     // cv::Mat clipped;
//     // cv::min(cv::max(u, cv::Scalar(0)), cv::Scalar(255), clipped);

//     // clipped.convertTo(result, CV_8U);
// }