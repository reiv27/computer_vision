#include "utils.hpp"

template <typename T>
inline T vmax(T first, T second) {
    return first > second ? first : second;
}

template <typename T, typename ... Args>
inline T vmax(T first, T second, Args ... tail) {
    return vmax(first > second ? first : second, tail...);
}


template <typename T>
inline T vmin(T first, T second) {
    return first < second ? first : second;
}

template <typename T, typename ... Args>
inline T vmin(T first, T second, Args ... tail) {
    return vmin(first < second ? first : second, tail...);
}


cv::Mat maxMinFilter(cv::Mat const& s) {
    int Ny = s.rows;
    int Nx = s.cols;
    cv::Mat d(Ny, Nx, CV_8U);

    for (int y = 0; y < Ny; ++ y) {
        int y0 = std::clamp(y-2, 0, Ny-1);
        int y1 = std::clamp(y-1, 0, Ny-1);
        int y2 = std::clamp(y,   0, Ny-1);
        int y3 = std::clamp(y+1, 0, Ny-1);
        int y4 = std::clamp(y+2, 0, Ny-1);

        uchar const* ps0 = s.ptr<uchar>(y0);
        uchar const* ps1 = s.ptr<uchar>(y1);
        uchar const* ps2 = s.ptr<uchar>(y2);
        uchar const* ps3 = s.ptr<uchar>(y3);
        uchar const* ps4 = s.ptr<uchar>(y4);
        uchar* pd = d.ptr<uchar>(y);

        for (int x = 0; x < Nx; ++ x) {
            int x0 = std::clamp(x-2, 0, Nx-1);
            int x1 = std::clamp(x-1, 0, Nx-1);
            int x2 = std::clamp(x,   0, Nx-1);
            int x3 = std::clamp(x+1, 0, Nx-1);
            int x4 = std::clamp(x+2, 0, Nx-1);

            pd[x] = vmax(
                ps0[x0], ps0[x1], ps0[x2], ps0[x3], ps0[x4],
                ps1[x0], ps1[x1], ps1[x2], ps1[x3], ps1[x4],
                ps2[x0], ps2[x1], ps2[x2], ps2[x3], ps2[x4],
                ps3[x0], ps3[x1], ps3[x2], ps3[x3], ps3[x4],
                ps4[x0], ps4[x1], ps4[x2], ps4[x3], ps4[x4]
            ) - vmin(
                ps0[x0], ps0[x1], ps0[x2], ps0[x3], ps0[x4],
                ps1[x0], ps1[x1], ps1[x2], ps1[x3], ps1[x4],
                ps2[x0], ps2[x1], ps2[x2], ps2[x3], ps2[x4],
                ps3[x0], ps3[x1], ps3[x2], ps3[x3], ps3[x4],
                ps4[x0], ps4[x1], ps4[x2], ps4[x3], ps4[x4]
            );
        }
    }
    return d;
}


cv::Mat fastMaxMinFilter(cv::Mat const& s) {
    int Ny = s.rows;
    int Nx = s.cols;
    cv::Mat d(Ny, Nx, CV_8U);

    uchar buf_max[Nx];
    uchar buf_min[Nx];

    for (int y = 0; y < Ny; ++y) {
        int y0 = std::clamp(y-2, 0, Ny-1);
        int y1 = std::clamp(y-1, 0, Ny-1);
        int y2 = std::clamp(y,   0, Ny-1);
        int y3 = std::clamp(y+1, 0, Ny-1);
        int y4 = std::clamp(y+2, 0, Ny-1);

        uchar const* ps0 = s.ptr<uchar>(y0);
        uchar const* ps1 = s.ptr<uchar>(y1);
        uchar const* ps2 = s.ptr<uchar>(y2);
        uchar const* ps3 = s.ptr<uchar>(y3);
        uchar const* ps4 = s.ptr<uchar>(y4);

        int x = 0;
        
        for (; x < Nx - 15; x += 16) {
            cv::v_uint8x16 u0 = cv::v_load(ps0 + x);
            cv::v_uint8x16 u1 = cv::v_load(ps1 + x);
            cv::v_uint8x16 u2 = cv::v_load(ps2 + x);
            cv::v_uint8x16 u3 = cv::v_load(ps3 + x);
            cv::v_uint8x16 u4 = cv::v_load(ps4 + x);
            cv::v_uint8x16 u = cv::v_max(cv::v_max(cv::v_max(cv::v_max(u0, u1), u2), u3), u4);
            cv::v_store(buf_max + x, u);
            u = cv::v_min(cv::v_min(cv::v_min(cv::v_min(u0, u1), u2), u3), u4);
            cv::v_store(buf_min + x, u);
        }

        for (; x < Nx; ++x) {
            buf_max[x] = vmax(ps0[x], ps1[x], ps2[x], ps3[x], ps4[x]);
            buf_min[x] = vmin(ps0[x], ps1[x], ps2[x], ps3[x], ps4[x]);
        }

        uchar* pd = d.ptr<uchar>(y);
        for (x = 0; x < Nx; ++x) {
            int x0 = std::clamp(x-2, 0, Nx-1);
            int x1 = std::clamp(x-1, 0, Nx-1);
            int x2 = std::clamp(x,   0, Nx-1);
            int x3 = std::clamp(x+1, 0, Nx-1);
            int x4 = std::clamp(x+2, 0, Nx-1);
            uint8_t max = vmax(buf_max[x0], buf_max[x1], buf_max[x2], buf_max[x3], buf_max[x4]);
            uint8_t min = vmin(buf_min[x0], buf_min[x1], buf_min[x2], buf_min[x3], buf_min[x4]);
            pd[x] = max - min;
        }
    }
    
    return d;
}