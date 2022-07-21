#include "ncnn/net.h"
#if defined(USE_NCNN_SIMPLEOCV)
#include "ncnn/simpleocv.h"
#else
#include "opencv2/opencv.hpp"
#endif

#include "seg.h"


struct Seg2D::CPrivate
{
	ncnn::Net net_;
	ncnn::Extractor ex_ = net_.create_extractor();
};

Seg2D::Seg2D() : mpD(new CPrivate)
{
	priorBox_ = nullptr;
	numPriors_ = 0;
}

Seg2D::~Seg2D()
{
	if (mpD) {
		mpD->ex_.clear();
		mpD->net_.clear();
		delete mpD;
	}
	if (priorBox_) {
		delete[] priorBox_;
		priorBox_ = nullptr;
	}
}

bool Seg2D::Build(std::string modelPath, const DetectOption& opt)
{
	// Load Model
	// mpD->net_.opt.use_vulkan_compute = true;

	FILE* fp = fopen(modelPath.c_str(), "rb");
	mpD->net_.load_param_bin(fp);
	mpD->net_.load_model(fp);
	fclose(fp);

	const std::vector<ncnn::Blob>& netBlobs = mpD->net_.blobs();
	const std::vector<ncnn::Layer*>& netLayers = mpD->net_.layers();
	std::cout << " blobs: " << netBlobs.size() << " layers: " << netLayers.size() << std::endl;

	mpD->ex_.clear();
	mpD->ex_ = mpD->net_.create_extractor();
	mpD->ex_.set_num_threads(4);

	// Get number of prior
	size_t numP = sizeof(opt.convW) / sizeof(opt.convW[0]);
	for (size_t p = 0; p < numP; ++p)
	{
		numPriors_ += opt.convH[p] * opt.convW[p] * 3;
	}

	// Generate prior box, ref make_prior() in yolact.py
	priorBox_ = new float[4 * numPriors_];
	float* pb = priorBox_;
	size_t numAR = sizeof(opt.aspectRatios) / sizeof(opt.aspectRatios[0]);
	for (size_t p = 0; p < numP; ++p) {
		int convW = opt.convW[p];
		int convH = opt.convH[p];
		float scale = opt.scales[p];

		// Iteration order is important(it has to sync up with the convout)
		for (int i = 0; i < convH; ++i) {
			for (int j = 0; j < convW; ++j) {
				// +0.5 because priors are in center-size notation
				float cx = (j + 0.5f) / convW;
				float cy = (i + 0.5f) / convH;

				for (size_t k = 0; k < numAR; ++k) {
					float ar = opt.aspectRatios[k];
					ar = sqrt(ar);
					float w = scale * ar / opt.targetSize;
					float h = scale / ar / opt.targetSize;

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

	return true;
}

bool Seg2D::IsValid() const
{
	/*if (!net_ || !priorBox_) {
		return false;
	}
	return true;*/
	if (!mpD || !priorBox_) {
		return false;
	}
	return true;
}

void Seg2D::Clear()
{
	if (mpD) {
		mpD->ex_.clear();
		mpD->net_.clear();
	}
}

struct Object
{
	cv::Rect_<float> rect;
	int label;
	float prob;
	std::vector<float> maskdata;
	cv::Mat mask;
};

void QsortDescentInplace(std::vector<float>& scores, int left, int right)
{
	int i = left;
	int j = right;
	float p = scores[(left + right) / 2];

	while (i <= j)
	{
		while (scores[i] > p)
			i++;

		while (scores[j] < p)
			j--;

		if (i <= j)
		{
			// swap
			std::swap(scores[i], scores[j]);

			i++;
			j--;
		}
	}

#pragma omp parallel sections
	{
#pragma omp section
		{
			if (left < j) QsortDescentInplace(scores, left, j);
		}
#pragma omp section
		{
			if (i < right) QsortDescentInplace(scores, i, right);
		}
	}
}

static void QsortDescentInplace(std::vector<float>& scores)
{
	if (scores.empty())
		return;

	QsortDescentInplace(scores, 0, scores.size() - 1);

}

static inline float IntersectionArea(const cv::Rect& a, const cv::Rect& b)
{
	cv::Rect_<float> inter = a & b;
	return inter.area();
}

void NMSForSortedBBoxes(const std::vector<cv::Rect> boxes, std::vector<int>& picked, float nmsThresh)
{
	picked.clear();
	const int n = boxes.size();
	std::vector<float> areas(n);
	for (int i = 0; i < n; ++i) {
		areas[i] = boxes[i].area();
	}

	for (int i = 0; i < n; ++i) {
		const cv::Rect box1 = boxes[i];

		int keep = 1;
		for (int j = 0; j < (int)picked.size(); ++j) {
			const cv::Rect box2 = boxes[picked[j]];

			float interArea = IntersectionArea(box1, box2);
			float unionArea = areas[i] + areas[picked[j]] - interArea;
			// IOU and other 
			if (interArea / unionArea > nmsThresh || interArea / areas[picked[j]] > 0.5 || interArea / areas[i] > 0.5) {
				keep = 0;
			}
		}
		if (keep) {
			picked.push_back(i);
		}
	}
}

template <typename T>
static inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
	const std::pair<float, T>& pair2)
{
	return pair1.first > pair2.first;
}

// Get max scores with corresponding indices.
	//    scores: a set of scores.
	//    scoreIndexVec: store the sorted (score, index) pair.
inline void GetMaxScoreIndex(const std::vector<float>& scores, std::vector<std::pair<float, int> >& scoreIndexVec)
{
	// Generate index score pairs.
	for (size_t i = 0; i < scores.size(); ++i)
	{
		scoreIndexVec.push_back(std::make_pair(scores[i], i));
	}

	// Sort the score pair according to the scores in descending order
	std::stable_sort(scoreIndexVec.begin(), scoreIndexVec.end(), SortScorePairDescend<int>);
}

void NMSBBoxes(const std::vector<cv::Rect> boxes, const std::vector<float>& scores, std::vector<int>& picked, float nmsThresh)
{
	// Sorted by score
	std::vector<std::pair<float, int> > scoreIndexVec;
	GetMaxScoreIndex(scores, scoreIndexVec);

	// Do nms.
	const int n = boxes.size();
	std::vector<float> areas(n);
	for (int i = 0; i < n; ++i) {
		areas[i] = boxes[i].area();
	}
	picked.clear();
	for (size_t i = 0; i < scoreIndexVec.size(); ++i) {
		const int idx = scoreIndexVec[i].second;
		bool keep = true;
		for (int k = 0; k < (int)picked.size() && keep; ++k) {
			const int keptIdx = picked[k];
			float interArea = IntersectionArea(boxes[idx], boxes[keptIdx]);
			float unionArea = areas[i] + areas[keptIdx] - interArea;
			if (interArea / unionArea > nmsThresh || interArea / areas[keptIdx] > 0.5 || interArea / areas[i] > 0.5) {
				keep = false;
			}
		}
		if (keep) {
			picked.push_back(idx);
		}

	}
}

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
	static const char* class_names[] = { "background", "tooth" };

	static const unsigned char colors[2][3] = { {56, 0, 255}, {226, 255, 0} };

	cv::Mat image = bgr.clone();

	int color_index = 0;

	for (size_t i = 0; i < objects.size(); i++)
	{
		const Object& obj = objects[i];

		if (obj.prob < 0.15)
			continue;

		fprintf(stderr, "%d class %d = %.5f at %.2f %.2f %.2f x %.2f\n", i + 1, obj.label, obj.prob,
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


bool Predict_(ncnn::Extractor& ex, float* priorBox_, int numPriors_, cv::Mat& srcImg, DetectRes& res, const DetectOption& opt)
{

	// 1. Preprocess img
#ifdef _DEBUG
	if (false) {
		cv::imshow("img", srcImg);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
#endif
	ncnn::Mat input = ncnn::Mat::from_pixels_resize(srcImg.data, ncnn::Mat::PIXEL_BGR2RGB, srcImg.cols, srcImg.rows, opt.targetSize, opt.targetSize);
	input.substract_mean_normalize(opt.MEANS, opt.STD);

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
	if (numPriors_ != confidence.h) {
		return false;
	}
	std::vector<std::vector<Object> > classCandidates;
	classCandidates.resize(numClass);
	for (int i = 0; i < numPriors_; ++i)
	{
		const float* conf = confidence.row(i);
		const float* loc = location.row(i);
		const float* pb = priorBox_ + i * 4;
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
		if (label == 0 || score <= opt.confThresh) {
			continue;
		}

		// center size
		float pbCx = pb[0];
		float pbCy = pb[1];
		float pbW = pb[2];
		float pbH = pb[3];

		float bboxCx = opt.var[0] * loc[0] * pbW + pbCx;
		float bboxCy = opt.var[1] * loc[1] * pbH + pbCy;
		float bboxW = (float)(exp(opt.var[2] * loc[2]) * pbW);
		float bboxH = (float)(exp(opt.var[3] * loc[3]) * pbH);

		float objX1 = bboxCx - bboxW * 0.5f;
		float objY1 = bboxCy - bboxH * 0.5f;
		float objX2 = bboxCx + bboxW * 0.5f;
		float objY2 = bboxCy + bboxH * 0.5f;

		// limit boundary
		objX1 = std::max(std::min(objX1 * srcImg.cols, (float)(srcImg.cols - 1)), 0.f);
		objY1 = std::max(std::min(objY1 * srcImg.rows, (float)(srcImg.rows - 1)), 0.f);
		objX2 = std::max(std::min(objX2 * srcImg.cols, (float)(srcImg.cols - 1)), 0.f);
		objY2 = std::max(std::min(objY2 * srcImg.rows, (float)(srcImg.rows - 1)), 0.f);

		// append object
		Object obj;
		obj.rect = cv::Rect_<float>(objX1, objY1, objX2 - objX1 + 1, objY2 - objY1 + 1);
		obj.label = label;
		obj.prob = score;
		obj.maskdata = std::vector<float>(maskData, maskData + mask.w);
		classCandidates[label].push_back(obj);
	}
	std::vector<Object> objects;
	for (int i = 0; i < classCandidates.size(); ++i) {
		std::vector<Object>& candidates = classCandidates[i];
		qsort_descent_inplace(candidates);
		std::vector<int> picked;
		nms_sorted_bboxes(candidates, picked, opt.nmsThresh);

		for (int j = 0; j < picked.size(); ++j) {
			int z = picked[j];
			objects.push_back(candidates[z]);
		}
	}
	qsort_descent_inplace(objects);

	// keep to k
	if (opt.topK < objects.size()) {
		objects.resize(opt.topK);
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
		cv::resize(mask, mask2, cv::Size(srcImg.cols, srcImg.rows));

		// crop obj box and binarize
		obj.mask = cv::Mat(srcImg.rows, srcImg.cols, CV_8UC1);
		{
			obj.mask = cv::Scalar(0);
			for (int y = 0; y < srcImg.rows; ++y) {
				if (y < obj.rect.y || y > obj.rect.y + obj.rect.height) {
					continue;
				}
				const float* mp2 = mask2.ptr<const float>(y);
				uchar* bmp = obj.mask.ptr<uchar>(y);
				for (int x = 0; x < srcImg.cols; ++x) {
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

	// draw_objects(srcImg, objects);
	const int contentSize = srcImg.cols * srcImg.rows;
	if (!res.bOverlap) {
		res.mask = new unsigned char[contentSize];
		for (int k = 0; k < contentSize; ++k) {
			res.mask[k] = 0;
		}
	}
	int n = 1;
	for (Object obj : objects) {
		res.boxes.push_back({ int(obj.rect.x), int(obj.rect.y), int(obj.rect.width), int(obj.rect.height) });
		res.scores.push_back(obj.prob);
		res.classIds.push_back(obj.label);
		if (res.bOverlap) {
			unsigned char* tmpMask = new unsigned char[contentSize];
			memcpy_s(tmpMask, contentSize, obj.mask.data, contentSize);
			res.masks.push_back(tmpMask);
		}
		else {
			unsigned char* resData = (unsigned char*)(res.mask);
			unsigned char* pData = (unsigned char*)(obj.mask.data);

			for (int row = 0; row < srcImg.rows; ++row) {
				for (int col = 0; col < srcImg.cols; ++col) {
					int k = row * srcImg.cols + col;
					if (pData[k] > 0) {
						resData[k] = n;
					}
				}
			}
			n += 1;

			// may be need find largest
		}

	}

	return true;
}


bool Seg2D::Predict(std::string imgPath, DetectRes& res, const DetectOption& opt)
{
	cv::Mat srcImg = cv::imread(imgPath);
	if (srcImg.empty()) {
		// read img failed
		return false;
	}

	return Predict_(mpD->ex_, priorBox_, numPriors_, srcImg, res, opt);

}

void Seg2D::Show(std::string imgPath, DetectRes& res, const DetectOption& opt)
{
	cv::Mat srcImg = cv::imread(imgPath);
	int imgW = srcImg.cols;
	int imgH = srcImg.rows;

	int n = res.boxes.size();

	for (int idx = 0; idx < n; ++idx) {
		std::vector<int> box = res.boxes[idx];
		int xmax = box[0] + box[2];
		int ymax = box[1] + box[3];

		// draw box
		cv::rectangle(srcImg, cv::Point(box[0], box[1]), cv::Point(xmax, ymax), cv::Scalar(0, 0, 255), 2);

		// draw class and score
		char text[256];
		sprintf(text, "%s: %.2f", opt.classNames[res.classIds[idx]].c_str(), res.scores[idx]);
		int baseLine;
		cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.2, 1, &baseLine);
		int ymin = std::max(box[1], labelSize.height);
		cv::putText(srcImg, text, cv::Point(box[0], ymin - baseLine), cv::FONT_HERSHEY_SIMPLEX, 0.25, cv::Scalar(0, 255, 0), 1);

		// draw mask
		if (res.bOverlap) {
			unsigned char* mask = res.masks[idx];
			for (int y = 0; y < imgH; y++)
			{
				const unsigned char* pmask = (unsigned char*)mask + y * imgW;
				uchar* p = srcImg.data + y * imgW * 3;
				for (int x = 0; x < imgW; x++)
				{
					if (pmask[x] > 0)
					{
						// (56, 94, 255) is color
						p[0] = (uchar)(p[0] * 0.5 + 56 * 0.5);
						p[1] = (uchar)(p[1] * 0.5 + 94 * 0.5);
						p[2] = (uchar)(p[2] * 0.5 + 255 * 0.5);
					}
					p += 3;
				}
			}
		}
	}
	if (!res.bOverlap) {
		for (int y = 0; y < imgH; y++)
		{
			const unsigned char* pmask = res.mask + y * imgW;
			uchar* p = srcImg.data + y * imgW * 3;
			for (int x = 0; x < imgW; x++)
			{
				if (pmask[x] > 0)
				{
					// (56, 94, 255) is color
					p[0] = (uchar)(p[0] * 0.5 + 56 * 0.5);
					p[1] = (uchar)(p[1] * 0.5 + 94 * 0.5);
					p[2] = (uchar)(p[2] * 0.5 + 255 * 0.5);
				}
				p += 3;
			}
		}
	}
	cv::imshow("img", srcImg);
	cv::waitKey(0);
	cv::destroyAllWindows();
	//cv::imwrite("D:/srcImg.png", srcImg);
}



