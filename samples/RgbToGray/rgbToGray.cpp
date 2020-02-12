#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <CUDASandbox/metric.hpp>
#include <cuda_runtime.h>

#include "kernels/rgbToGray.cuh"

cv::Mat rgb_to_gray_cpu(const cv::Mat& rgb_img, metric& metric)
{
	metric.start(metric_type::CALCULATION);
	
	cv::Mat gray_img;
	gray_img.create(rgb_img.rows, rgb_img.cols, CV_8UC1);

	for (auto i = 0; i < rgb_img.rows; i++)
	{
		for (auto j = 0; j < rgb_img.cols; j++)
		{
			const auto b = rgb_img.at<cv::Vec3b>(i, j)[0] * .114F;
			const auto g = rgb_img.at<cv::Vec3b>(i, j)[1] * .587F;
			const auto r = rgb_img.at<cv::Vec3b>(i, j)[2] * .299F;

			gray_img.at<uint8_t>(i, j) = r + g + b;
		}
	}

	metric.stop(metric_type::CALCULATION);
	return gray_img;
}

cv::Mat rgb_to_gray_gpu(const cv::Mat& rgb_img, metric& metric)
{
	cv::Mat gray_img;
	gray_img.create(rgb_img.rows, rgb_img.cols, CV_8UC1);

	const auto* rgb_img_ptr = rgb_img.ptr<uchar3>(0);
	auto gray_img_ptr = gray_img.ptr<uint8_t>(0);

	gpu::rgb_to_gray(rgb_img_ptr, gray_img_ptr, rgb_img.rows, rgb_img.cols, metric);

	return gray_img;
}

int main()
{
    const auto img = cv::imread("assets/island.jpg", cv::IMREAD_COLOR);
	
    cv::namedWindow("Original", cv::WINDOW_KEEPRATIO);
	cv::imshow("Original", img);
	cv::resizeWindow("Original", 1280, 720);

	printf("[CPU] Converting image to gray scale...\n");
	metric cpu_metric(img.rows * img.cols);
	const auto cpu_gray = rgb_to_gray_cpu(img, cpu_metric);
	cpu_metric.serialize("CPU Computation metrics");
	
	printf("[GPU] Converting image to gray scale...\n");
	metric gpu_metric(img.rows * img.cols);
	const auto gpu_gray = rgb_to_gray_gpu(img, gpu_metric);
	gpu_metric.serialize("GPU Computation metrics");

	cv::namedWindow("Gray scale", cv::WINDOW_KEEPRATIO);
	cv::imshow("Gray scale", gpu_gray);
	cv::resizeWindow("Gray scale", 1280, 720);

    cv::waitKey();
    return 0;
}