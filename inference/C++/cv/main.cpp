#include <chrono>
#include <iostream>
#include "yolact.h"


int main(int argc, char* argv[])
{
	
	std::string modelPath = "D:/ncnn/ncnn_install/Release/ncnn_install_no_vulkan/bin/test_sim.onnx";  // "E:/code/Server223/yolact/inference/tooth_754_80000.onnx";
	std::string imgPath = "E:/code/Server223/yolact/inference/test.png";

	Yolact yolact;
	const YolactDetectOption opt;
	yolact.Build(modelPath, opt);
	YolactDetectRes res;

	yolact.Predict(imgPath, res, opt);  // skip first
	auto start = std::chrono::system_clock::now();
	for(int i = 0; i < 100; ++i){
		yolact.Predict(imgPath, res, opt);
	}
	auto end = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << " s!" << std::endl;

	yolact.Show(imgPath, res, opt);

	return 0;
}