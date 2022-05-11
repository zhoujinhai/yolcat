#include "yolact.h"


int main(int argc, char* argv[])
{
	
	std::string modelPath = "E:/code/Server223/yolact/inference/tooth_754_80000.onnx";
	std::string imgPath = "E:/code/Server223/yolact/inference/test.png";

	Yolact yolact;
	const YolactDetectOption opt;
	yolact.Build(modelPath, opt);
	YolactDetectRes res;
	yolact.Predict(imgPath, res, opt);

	yolact.Show(imgPath, res, opt);

	return 0;
}