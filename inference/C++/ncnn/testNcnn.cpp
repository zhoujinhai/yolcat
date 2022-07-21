#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>

#include "ncnn/net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "ncnn/simpleocv.h"
#else
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#endif

#include "seg.h"


struct Object
{
	cv::Rect_<float> rect;
	int label;
	float prob;
	std::vector<float> maskdata;
	cv::Mat mask;
};

static inline float intersection_area(const Object& a, const Object& b)
{
	cv::Rect_<float> inter = a.rect & b.rect;
	return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
	int i = left;
	int j = right;
	float p = objects[(left + right) / 2].prob;

	while (i <= j)
	{
		while (objects[i].prob > p)
			i++;

		while (objects[j].prob < p)
			j--;

		if (i <= j)
		{
			// swap
			std::swap(objects[i], objects[j]);

			i++;
			j--;
		}
	}

#pragma omp parallel sections
	{
#pragma omp section
		{
			if (left < j) qsort_descent_inplace(objects, left, j);
		}
#pragma omp section
		{
			if (i < right) qsort_descent_inplace(objects, i, right);
		}
	}
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
	if (objects.empty())
		return;

	qsort_descent_inplace(objects, 0, objects.size() - 1);

}


static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold)
{
	picked.clear();

	const int n = objects.size();

	std::vector<float> areas(n);
	for (int i = 0; i < n; i++)
	{
		areas[i] = objects[i].rect.area();
	}

	for (int i = 0; i < n; i++)
	{
		const Object& a = objects[i];

		int keep = 1;
		for (int j = 0; j < (int)picked.size(); ++j)
		{
			const Object& b = objects[picked[j]];

			// intersection over union
			float inter_area = intersection_area(a, b);
			float union_area = areas[i] + areas[picked[j]] - inter_area;
			// float IoU = inter_area / union_area
			if (inter_area / union_area > nms_threshold || inter_area / areas[picked[j]] > 0.5 || inter_area / areas[i] > 0.5)
				keep = 0;
		}

		if (keep)
			picked.push_back(i);
	}
}


static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
	static const char* class_names[] = { "background", "tooth"};

	static const unsigned char colors[2][3] = {{56, 0, 255}, {226, 255, 0}};

	cv::Mat image = bgr.clone();

	int color_index = 0;

	for (size_t i = 0; i < objects.size(); i++)
	{
		const Object& obj = objects[i];

		if (obj.prob < 0.15)
			continue;

		fprintf(stderr, "%d class %d = %.5f at %.2f %.2f %.2f x %.2f\n", i+1, obj.label, obj.prob,
			obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

		const unsigned char* color = colors[color_index % 81];
		color_index++;

		cv::rectangle(image, obj.rect, cv::Scalar(color[0], color[1], color[2]));

		char text[256];
		sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

		int baseLine = 0;
		cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

		int x = obj.rect.x;
		int y = obj.rect.y - label_size.height - baseLine;
		if (y < 0)
			y = 0;
		if (x + label_size.width > image.cols)
			x = image.cols - label_size.width;

		cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
			cv::Scalar(255, 255, 255), -1);

		cv::putText(image, text, cv::Point(x, y + label_size.height),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

		// draw mask
		for (int y = 0; y < image.rows; y++)
		{
			const uchar* mp = obj.mask.ptr(y);
			uchar* p = image.ptr(y);
			for (int x = 0; x < image.cols; x++)
			{
				if (mp[x] == 255)
				{
					p[0] = cv::saturate_cast<uchar>(p[0] * 0.5 + color[0] * 0.5);
					p[1] = cv::saturate_cast<uchar>(p[1] * 0.5 + color[1] * 0.5);
					p[2] = cv::saturate_cast<uchar>(p[2] * 0.5 + color[2] * 0.5);
				}
				p += 3;
			}
		}
	}

	cv::imwrite("result.png", image);
	cv::imshow("image", image);
	cv::waitKey(0);
}


static bool predict(ncnn::Extractor& ex, const cv::Mat& img, std::vector<Object>& objects)
{
	objects.clear();

	// 1. Preprocess img
#ifdef _DEBUG
	if (false) {
		cv::imshow("img", img);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
#endif

	const int targetSize = 550;
	ncnn::Mat input = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, targetSize, targetSize);
	const float mean_vals[3] = { 123.68f, 116.78f, 103.94f };
	const float norm_vals[3] = { 1.0 / 58.40f, 1.0 / 57.12f, 1.0 / 57.38f };
	input.substract_mean_normalize(mean_vals, norm_vals);

	// 2. Inference
	ex.input(0, input);   // set input  "img" ==> test_sim_param_id::BLOB_img = 0
	// get output
	ncnn::Mat maskMaps;
	ncnn::Mat location;
	ncnn::Mat mask;
	ncnn::Mat confidence;

	ex.extract(276, maskMaps);  // 138 * 138 * 32      "355" ==> "proto" ==> test_sim_param_id::BLOB_proto = 276  后面文件不需要引号
	ex.extract(273, location);    // 4 * 19248         "551" ==> "loc" ==> test_sim_param_id::BLOB_loc = 273
	ex.extract(275, mask);       // 32 * 19248         "553" ==> "mask" ==> test_sim_param_id::BLOB_mask = 275
	ex.extract(274, confidence); // n_class * 19248    "554" ==> "conf" ==> test_sim_param_id::BLOB_conf = 274

	// 3. Parse result
	int numClass = confidence.w;
	int numPriors = confidence.h;

	// some super param
	const int convW[5] = { 69, 35, 18, 9, 5 };
	const int convH[5] = { 69, 35, 18, 9, 5 };
	const float aspectRatios[3] = { 1.f, 0.5f, 2.f };
	const float scales[5] = { 24.f, 48.f, 96.f, 192.f, 384.f };
	const float var[4] = { 0.1f, 0.1f, 0.2f, 0.2f };
	const float confThresh = 0.55;
	const float nmsThresh = 0.5;
	const int topK = 20;

	// make priorbox
	ncnn::Mat priorBox(4, numPriors);
	{
		float* pb = priorBox;
		for (int p = 0; p < 5; ++p) {
			for (int i = 0; i < convH[p]; ++i) {
				for (int j = 0; j < convW[p]; ++j) {
					// +0.5 because priors are in center-size notation
					float cx = (j + 0.5f) / convW[p];
					float cy = (i + 0.5f) / convH[p];

					for (int k = 0; k < 3; ++k) {
						float ar = sqrt(aspectRatios[k]);
						float w = scales[p] * ar / targetSize;
						float h = scales[p] / ar / targetSize;

						// This is for backward compatability with a bug where I made everything square by accident
						// if cfg.backbone.use_square_anchors:
						h = w;

						pb[0] = cx;
						pb[1] = cy;
						pb[2] = w;
						pb[3] = h;
						pb += 4;
					}
				}
			}
		}
	}

	std::vector<std::vector<Object> > classCandidates;
	classCandidates.resize(numClass);
	for (int i = 0; i < numPriors; ++i)
	{
		const float* conf = confidence.row(i);
		const float* loc = location.row(i);
		const float* pb = priorBox.row(i);
		const float* maskData = mask.row(i);

		// find class id with highest score, start from 1 to skip background
		int label = 0;
		float score = 0.f;
		for (int j = 1; j < numClass; ++j) {
			float classScore = conf[j];
			if (classScore > score) {
				label = j;
				score = classScore;
			}
		}

		// ignore background or low score
		if (label == 0 || score <= confThresh) {
			continue;
		}

		// center size
		float pbCx = pb[0];
		float pbCy = pb[1];
		float pbW = pb[2];
		float pbH = pb[3];

		float bboxCx = var[0] * loc[0] * pbW + pbCx;
		float bboxCy = var[1] * loc[1] * pbH + pbCy;
		float bboxW = (float)(exp(var[2] * loc[2]) * pbW);
		float bboxH = (float)(exp(var[3] * loc[3]) * pbH);

		float objX1 = bboxCx - bboxW * 0.5f;
		float objY1 = bboxCy - bboxH * 0.5f;
		float objX2 = bboxCx + bboxW * 0.5f;
		float objY2 = bboxCy + bboxH * 0.5f;

		//  limit boundary
		objX1 = std::max(std::min(objX1 * img.cols, (float)(img.cols - 1)), 0.f);
		objY1 = std::max(std::min(objY1 * img.rows, (float)(img.rows - 1)), 0.f);
		objX2 = std::max(std::min(objX2 * img.cols, (float)(img.cols - 1)), 0.f);
		objY2 = std::max(std::min(objY2 * img.rows, (float)(img.rows - 1)), 0.f);

		// append object
		Object obj;
		obj.rect = cv::Rect_<float>(objX1, objY1, objX2 - objX1 + 1, objY2 - objY1 + 1);
		obj.label = label;
		obj.prob = score;
		obj.maskdata = std::vector<float>(maskData, maskData + mask.w);
		classCandidates[label].push_back(obj);

	}

	for (int i = 0; i < classCandidates.size(); ++i) {
		std::vector<Object>& candidates = classCandidates[i];
		qsort_descent_inplace(candidates);
		std::vector<int> picked;
		nms_sorted_bboxes(candidates, picked, nmsThresh);

		for (int j = 0; j < picked.size(); ++j) {
			int z = picked[j];
			objects.push_back(candidates[z]);
		}
	}
	qsort_descent_inplace(objects);

	// keep to k
	if (topK < objects.size()) {
		objects.resize(topK);
	}

	// generate mask
	for (int i = 0; i < objects.size(); ++i) {
		Object& obj = objects[i];
		cv::Mat mask(maskMaps.h, maskMaps.w, CV_32FC1);
		{
			mask = cv::Scalar(0.f);
			for (int p = 0; p < maskMaps.c; ++p) {
				const float* maskMap = maskMaps.channel(p);
				float coeff = obj.maskdata[p];
				float* mp = (float*)mask.data;

				// mask += m * coeff
				for (int j = 0; j < maskMaps.w * maskMaps.h; ++j) {
					mp[j] += maskMap[j] * coeff;
				}
			}
		}

		cv::Mat mask2;
		cv::resize(mask, mask2, cv::Size(img.cols, img.rows));

		// crop obj box and binarize
		obj.mask = cv::Mat(img.rows, img.cols, CV_8UC1);
		{
			obj.mask = cv::Scalar(0);
			for (int y = 0; y < img.rows; ++y) {
				if (y < obj.rect.y || y > obj.rect.y + obj.rect.height) {
					continue;
				}
				const float* mp2 = mask2.ptr<const float>(y);
				uchar* bmp = obj.mask.ptr<uchar>(y);
				for (int x = 0; x < img.cols; ++x) {
					if (x < obj.rect.x || x > obj.rect.x + obj.rect.width) {
						continue;
					}
					bmp[x] = mp2[x] > 0.5f ? 255 : 0;
				}
			}
		}
	}

	maskMaps.release();
	confidence.release();
	location.release();
	mask.release();

	return true;
}


int main(int argc, char* argv[])
{
	const std::string paramPath = "E:/code/TestC++/ncnn/test_sim.param";
	const std::string binPath = "E:/code/TestC++/ncnn/test_sim.bin";
	const std::string paramBinPath = "E:/code/TestC++/ncnn/test_sim.param.bin";
	const std::string allBin = "E:/code/TestC++/ncnn/test_sim_all.bin";  // cat test_sim.param.bin test_sim.bin > test_sim_all.bin
	
	const std::string testImgPath = "E:/code/TestC++/ncnn/top.png";

	Seg2D seg2D;
	DetectOption opt;
	DetectRes res;
	seg2D.Build(allBin, opt);
	if (seg2D.IsValid()) {
		seg2D.Predict(testImgPath, res, opt);   // skip first
		auto start = std::chrono::system_clock::now();
		for (int i = 0; i < 100; ++i) {
			seg2D.Predict(testImgPath, res, opt);
		}
		auto end = std::chrono::system_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		std::cout << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << " s!" << std::endl;
	}
	seg2D.Show(testImgPath, res, opt);
	seg2D.Clear();
	return 0;

	// Load Model
	ncnn::Net net;

	// net.opt.use_vulkan_compute = true;
	////// 1.Load model
	//// --- method 1 ---
	//net.load_param(paramPath.c_str());
	//net.load_model(binPath.c_str());

	//// --- method 2 ---
	//net.load_param_bin(paramBinPath.c_str());
	//net.load_model(binPath.c_str());

	//// --- method 3 ---
	// // need #inlucde "test_sim.mem.h"
	//net.load_param(test_sim_param_bin);
	//net.load_model(test_sim_bin);

	// --- method 4 ---
	FILE* fp = fopen(allBin.c_str(), "rb");
	int a = net.load_param_bin(fp);
	int b = net.load_model(fp);
	fclose(fp);

	const std::vector<ncnn::Blob>& netBlobs = net.blobs();
	const std::vector<ncnn::Layer*>& netLayers = net.layers();
	std::cout << " blobs: " << netBlobs.size() << " layers: " << netLayers.size() << std::endl;


	// Load img
	cv::Mat img = cv::imread(testImgPath);
#ifdef _DEBUG
	if (false) {
		cv::imshow("img", img);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
#endif

	std::vector<Object> objects;

	ncnn::Extractor ex = net.create_extractor();
	ex.set_num_threads(4);

	auto start1 = std::chrono::system_clock::now();
	for (int i = 0; i < 1; ++i) {
		predict(ex, img, objects);
	}
	auto end1 = std::chrono::system_clock::now();
	auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
	std::cout << double(duration1.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << " s!" << std::endl;

	draw_objects(img, objects);

	ex.clear();
	net.clear();
	return 0;
}