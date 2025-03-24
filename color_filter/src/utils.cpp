#include "inc/utils.hpp"


void createMask(const cv::Mat& bad_mask, cv::Mat& mask) {
    int Nx = bad_mask.cols;
    int Ny = bad_mask.rows;

    for (int y = 0; y < Ny; ++y) {
        cv::Vec3b const* ps = bad_mask.ptr<cv::Vec3b>(y);
        uchar* pd = mask.ptr<uchar>(y);

        for (int x = 0; x < Nx; ++x) {
            if ((*ps)[0] != 0 || (*ps)[1] != 0 || (*ps)[2] != 0) {
                *pd = 255;
            }
            ++ps;
            ++pd;
        }
    }
}


void createFilter(const cv::Mat& src, const cv::Mat& mask, cv::Vec3f& p0, cv::Vec3f& v, float& t_min, float& t_max, float& R) {
    int Ny = src.rows;
    int Nx = src.cols;    

    int n = 0;
    std::vector<cv::Vec3f> pts;
    pts.reserve(n);
    for (int y = 0; y < Ny; ++y) {
        uchar const* ps = mask.ptr<uchar>(y);
        cv::Vec3b const* pd = src.ptr<cv::Vec3b>(y);
        for (int x = 0; x < Nx; ++x) {
            if (*ps != 0) {
                pts.push_back(*pd);
                n++;
            }
            ++ps;
            ++pd;
        }
    }

    p0 = cv::Vec3f(0, 0, 0); // B G R
    for (int i = 0; i < n; ++i) {
        p0 += pts[i];
    }
    p0 /= n;
   
    std::vector<cv::Vec3f> d;
    d.reserve(n);
    for (int i = 0; i < n; ++i) {
        d.push_back(pts[i] - p0);
    }

    // Calculating eigenvectors and v
    cv::Matx33f D = cv::Matx33f::zeros(); 
    for (int i = 0; i < n; ++i) {
        D += d[i] * d[i].t();
    }

    cv::Vec3f eigenvalues;
    cv::Mat eigenvectors;
    cv::eigen(D, eigenvalues, eigenvectors);

    float max_eigenvalue = std::max(eigenvalues[0], std::max(eigenvalues[1], eigenvalues[3]));
    v = cv::Vec3f(eigenvectors.at<float>(0, 0), eigenvectors.at<float>(1, 0), eigenvectors.at<float>(2, 0));
    float v_norm = cv::norm(v);
    float v_norm_2 = v_norm * v_norm;

    // Calculating t, t_min, t_max
    cv::Mat t(n, 1, CV_32F, cv::Scalar(0.0));
    for (int i = 0; i < n; ++i) {
        t.at<float>(i) = ((pts[i] - p0).t() * v)[0] / v_norm_2;
    }
    cv::sort(t, t, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
    float min_board = 0.001;
    float max_board = 0.999;
    t_min = t.at<float>(int(n*min_board));
    t_max = t.at<float>(int(n*max_board));

    // Calculating R
    cv::Mat r(n, 1, CV_32F, cv::Scalar(0.0));
    for (int i = 0; i < n; ++i) {
        r.at<float>(i) = cv::norm((pts[i] - p0).cross(v)) / v_norm_2;
    }
    cv::sort(r, r, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
    float max_radius = 0.95;
    R = r.at<float>(int(n*max_radius));
}