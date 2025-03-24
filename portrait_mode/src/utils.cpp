#include "utils.hpp"


cv::Mat enhance(cv::Mat const& f, cv::Mat const& w, float lam) {
    cv::Mat f_float;
    f.convertTo(f_float, CV_32F);
    u_int16_t Ny = f_float.rows;
    u_int16_t Nx = f_float.cols;
    
    // Finding grad_f
    cv::Mat grad_fx(Ny, Nx, CV_32F);
    cv::Mat grad_fy(Ny, Nx, CV_32F);
    cv::Sobel(f_float, grad_fx, CV_32F, 1, 0);
    cv::Sobel(f_float, grad_fy, CV_32F, 0, 1);

    // Finding gx, gy
    cv::Mat gx(Ny, Nx, CV_32FC2, cv::Scalar(0,0));
    cv::Mat gy(Ny, Nx, CV_32FC2, cv::Scalar(0,0));
    cv::Mat complex_f(Ny, Nx, CV_32FC2, cv::Scalar(0,0));
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            gx.at<cv::Vec2f>(y, x)[0] = grad_fx.at<float>(y, x) * w.at<uchar>(y, x) / 100.0;
            gy.at<cv::Vec2f>(y, x)[0] = grad_fy.at<float>(y, x) * w.at<uchar>(y, x) / 100.0;
            complex_f.at<cv::Vec2f>(y, x)[0] = f_float.at<float>(y, x);
        }
    }

    // Furie
    cv::Mat Gx(Ny, Nx, CV_32FC2);
    cv::Mat Gy(Ny, Nx, CV_32FC2);
    cv::Mat F(Ny, Nx, CV_32FC2);
    cv::dft(gx, Gx);
    cv::dft(gy, Gy);
    cv::dft(complex_f, F);

    // D
    cv::Mat Dx(Ny, Nx, CV_32FC2, cv::Scalar(0, 0));
    cv::Mat Dy(Ny, Nx, CV_32FC2, cv::Scalar(0, 0));

    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            float real_part = 0.0f;
            float imag_part = std::sin(2 * M_PI * x / static_cast<float>(Nx));
            Dx.at<cv::Vec2f>(y, x) = cv::Vec2f(real_part, imag_part);
            real_part = 0.0f;
            imag_part = std::sin(2 * M_PI * y / static_cast<float>(Ny));
            Dy.at<cv::Vec2f>(y, x) = cv::Vec2f(real_part, imag_part);
        }
    }

    cv::Mat Mult1(Ny, Nx, CV_32FC2, cv::Scalar(0, 0));
    cv::Mat Mult2(Ny, Nx, CV_32FC2, cv::Scalar(0, 0));
    cv::Mat Ones(Ny, Nx, CV_32FC2, cv::Scalar(1, 0));

    cv::mulSpectrums(Dx, Gx, Mult1, 0);
    cv::mulSpectrums(Dy, Gy, Mult2, 0);
    cv::Mat num = -F + lam * Mult1 + lam * Mult2;        

    cv::mulSpectrums(Dx, Dx, Mult1, 0); 
    cv::mulSpectrums(Dy, Dy, Mult2, 0);
    cv::Mat den = lam * Mult1 + lam * Mult2 - Ones;

    cv::Mat U(Ny, Nx, CV_32FC2, cv::Scalar(0, 0));
    cv::divSpectrums(num, den, U, 0);

    cv::Mat u;
    cv::idft(U, u, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    u.convertTo(u, CV_8U);
    
    return u;
}