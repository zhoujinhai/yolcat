#include <string>
#include <vector>

class YolactDetectOption {
public:
	const std::vector<std::string> classNames = {"background", "teeth"};
	const int targetSize = 550;
	const float MEANS[3] = { 123.68, 116.78, 103.94 };
	const float STD[3] = { 58.40, 57.12, 57.38 };
	const int convW[5] = { 69, 35, 18, 9, 5 };                      // the last feature size after FPN
	const int convH[5] = { 69, 35, 18, 9, 5 };
	const float aspectRatios[3] = { 1.f, 0.5f, 2.f };
	const float scales[5] = { 24.f, 48.f, 96.f, 192.f, 384.f };
	const float var[4] = { 0.1f, 0.1f, 0.2f, 0.2f };
	const int maskH = 138;
	const int maskW = 138;
	
	float confThresh = 0.8;  
	float nmsThresh = 0.2;
	int topK = 100;
};


class YolactDetectRes {
public:
	std::vector<std::vector<int>> boxes;    // left top x,y  and w , h
	std::vector<double> scores;
	std::vector<int> classIds;
	std::vector<unsigned char*> masks;      // if no overlap region, it can just save one mask
	bool bOverlap = false;                  // if true, everyone has ptr, false, just save one
	unsigned char* mask;

	~YolactDetectRes() {
		if (masks.size() > 0) {
			for (auto& m : masks) {
				delete[] m;
				m = nullptr;
			}
			masks.clear();
		}
		boxes.clear();
		scores.clear();
		classIds.clear();
		if (mask) {
			delete[] mask;
			mask = nullptr;
		}
	};
};

class Yolact {
public:

	Yolact();
	~Yolact();

	/*
	* @brief: load the yolact model that has been conveted to onnx format
	* 
	* @param[in] modelPath: the file path of model
	* @param[in] opt: some params for yolact net
	*
	*/
	bool Build(std::string modelPath, const YolactDetectOption& opt);

	/*
	* @brief: predict the img
	*
	* @param[in] imgPath: the file path of image
	* @param[out] res: the predict result 
	* @param[in] opt: some params for predict and network
	*
	*/
	bool Predict(std::string imgPath, YolactDetectRes & res, const YolactDetectOption& opt);

	/*
	* @brief: show the predict res
	*/
	void Show(std::string imgPath, YolactDetectRes& res, const YolactDetectOption& opt);


private:
	void* net_;
	float* priorBox_;
	int numPriors_;

};