#include "yolact.h"
#include "opencv2/opencv.hpp"


Yolact::Yolact()
{
	net_ = nullptr;
	priorBox_ = nullptr;
	numPriors_ = 0;
}

Yolact::~Yolact()
{
	if (net_) {
		delete static_cast<cv::dnn::Net *>(net_);
		net_ = nullptr;
	}
	if (priorBox_) {
		delete priorBox_;
		priorBox_ = nullptr;
	}
}

bool Yolact::Build(std::string modelPath, const YolactDetectOption& opt)
{
	// Load model
	cv::dnn::Net *net = new cv::dnn::Net();
	*net = cv::dnn::readNetFromONNX(modelPath);
	if (net->empty()) {
		// load model failed
		return false;
	}
	net_ = net;

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

bool PreProcess(const cv::Mat& srcImg, cv::Mat& img, const YolactDetectOption& opt)
{
	// resize
	cv::resize(srcImg, img, cv::Size(opt.targetSize, opt.targetSize), cv::INTER_LINEAR);
	
	// convert channels of RGB
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

	// Normalize
	img.convertTo(img, CV_32F);
	for (int r = 0; r < img.rows; ++r) {
		float* pData = (float*)(img.data + r * img.step);
		for (int c = 0; c < img.cols; ++c) {
			for (int i = 0; i < 3; ++i) {
				pData[i] = (pData[i] - opt.MEANS[i]) / opt.STD[i];
			}
			pData += 3;
		}
	}
	return true;
}

void Sigmoid(cv::Mat& img, int size) {
	float* pData = (float *)(img.data);
	for (int i = 0; i < size; ++i) {
		pData[i] = 1.0 / (1 + expf(-pData[i]));
	}
}

void LimitRegion(cv::Mat& img, cv::Rect box, int n, float value = 0.5)
{
	// img has 1 channel
	int w = img.cols;
	int h = img.rows;
	float* pData = (float*)(img.data);

	float leftTopX = box.x;
	float leftTopY = box.y;
	float rightBottomX = box.x + box.width;
	float rightBottomY = box.y + box.height;

	for (int row = 0; row < h; ++row) {
		for (int col = 0; col < w; ++col) {
			int idx = row * w + col;
			if (col < leftTopX || col > rightBottomX || row < leftTopY || row > rightBottomY) {
				pData[idx] = 0;
			}
			else {
				if (pData[idx] < value) {
					pData[idx] = 0;
				}
				else {
					pData[idx] = n;
				}
			}
		}
	}
	
}

bool Yolact::Predict(std::string imgPath,YolactDetectRes& res, const YolactDetectOption& opt)
{
	if (!net_ || !priorBox_) {
		return false;
	}
	cv::dnn::Net* net = static_cast<cv::dnn::Net*>(net_);
	cv::Mat srcImg = cv::imread(imgPath);
	if (srcImg.empty()) {
		// read img failed
		return false;
	}

	int imgW = srcImg.cols;
	int imgH = srcImg.rows;

	cv::Mat img;
	if (!PreProcess(srcImg, img, opt)) {
		return false;
	}

	// predict, 
    //loc:   numPriors_ * 4
	//conf:  numPriors_ * classNumber 
	//mask:  numPriors_ * 32
	//proto: maskW * maskH * 32
	std::vector<cv::Mat> predictRes;
	cv::Mat blob = cv::dnn::blobFromImage(img);
	net->setInput(blob);
	net->forward(predictRes, net->getUnconnectedOutLayersNames());

	// parse res
	std::vector<int> classIds;
	std::vector<float> confs;
	std::vector<cv::Rect> boxes;
	std::vector<int> maskIds;
	const int numClass = predictRes[1].cols;
	for (int i = 0; i < numPriors_; ++i) {
		cv::Mat conf = predictRes[1].row(i).colRange(0, numClass);
		cv::Point classIdPt;
		double score;
		cv::minMaxLoc(conf, 0, &score, 0, &classIdPt);  // class
		if (classIdPt.x > 0 && score > opt.confThresh) {
			const float* loc = (float*)predictRes[0].data + i * 4;
			const float* pb = priorBox_ + i * 4;
			float cx = pb[0];
			float cy = pb[1];
			float w = pb[2];
			float h = pb[3];

			float bboxCX = opt.var[0] * loc[0] * w + cx;     // box, ref function Detect in yolact.py
			float bboxCY = opt.var[1] * loc[1] * h + cy;
			float bboxW = (float)(exp(opt.var[2] * loc[2]) * w);
			float bboxH = (float)(exp(opt.var[3] * loc[3]) * h);

			float bboxX1 = bboxCX - bboxW * 0.5f;         
			float bboxY1 = bboxCY - bboxH * 0.5f;
			float bboxX2 = bboxCX + bboxW * 0.5f;
			float bboxY2 = bboxCY + bboxH * 0.5f;

			// limit boundary
			bboxX1 = std::max(std::min(bboxX1 * imgW, (float)(imgW - 1)), 0.f);
			bboxY1 = std::max(std::min(bboxY1 * imgH, (float)(imgH - 1)), 0.f);
			bboxX2 = std::max(std::min(bboxX2 * imgW, (float)(imgW - 1)), 0.f);
			bboxY2 = std::max(std::min(bboxY2 * imgH, (float)(imgH - 1)), 0.f);

			// save res
			classIds.push_back(classIdPt.x);
			confs.push_back(score);
			boxes.push_back(cv::Rect(int(bboxX1), int(bboxY1), int(bboxX2 - bboxX1 + 1), int(bboxY2 - bboxY1 + 1)));   // left top with w / h
			maskIds.push_back(i);
		}
	}

	// NMS
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confs, opt.confThresh, opt.nmsThresh, indices, 1.f, opt.topK);

	// get result
	int n = 1;
	const int contentSize = imgW * imgH;
	if (!res.bOverlap) {
		res.mask = new unsigned char[contentSize];
		for (int k = 0; k < contentSize; ++k) {
			res.mask[k] = 0;
		}
	}
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];

		// mask
		cv::Mat mask = cv::Mat::zeros(opt.maskH, opt.maskW, CV_32FC1);

		const float* protoMap = (float*)predictRes[3].data;
		int channel = predictRes[2].cols;
		int area = opt.maskH * opt.maskW;
		float* coeff = (float*)predictRes[2].data + maskIds[idx] * channel;

		// masks = proto_data @ masks.t()
		float* pMask = (float*)mask.data;
		for (int j = 0; j < area; ++j) {
			for (int c = 0; c < channel; ++c) {
				pMask[j] += protoMap[c] * coeff[c];
			}
			protoMap += channel;
		}

		Sigmoid(mask, area);

		cv::resize(mask, mask, cv::Size(imgW, imgH));

		// Use bbox to limit mask region  // masks = crop(masks, boxes)  binarize mask   // masks.gt_(0.5)
		LimitRegion(mask, box, n);

		//unsigned char* maskData = mask.data;
		res.boxes.push_back({ box.x, box.y, box.width, box.height });
		res.scores.push_back(confs[idx]);
		res.classIds.push_back(classIds[idx]);

		mask.convertTo(mask, CV_8UC1);
		if (res.bOverlap) {
			unsigned char* tmpMask = new unsigned char[contentSize];
			memcpy_s(tmpMask, contentSize, mask.data, contentSize);
			res.masks.push_back(tmpMask);
		}
		else {
			unsigned char* resData = (unsigned char*)(res.mask);
			unsigned char* pData = (unsigned char*)(mask.data);

			/*for (int row = 0; row < imgH; ++row) {
				for (int col = 0; col < imgW; ++col) {
					int k = row * imgW + col;
					if (pData[k] > 0) {
						resData[k] = n;
					}
				}
			}*/
			
			// find largest 
			std::vector<std::vector < std::vector<int> >> components;
			std::queue<std::vector<int> > pixelQueue;
			for (int i = 0; i < imgW; ++i)
			{
				for (int j = 0; j < imgH; ++j)
				{
					unsigned char* pPixel = pData + (j * imgW + i);
					if (*pPixel > 0)
					{
						std::vector < std::vector<int> > currentComponent;
						currentComponent.push_back({ i, j });
						pixelQueue.push({ i, j });
						*pPixel = 0;
						while (!pixelQueue.empty())
						{
							std::vector<int> currentPixel = pixelQueue.front();
							pixelQueue.pop();

							for (int k = -1; k <= 1; ++k)
							{
								int x = currentPixel[0] + k;
								if (x >= 0 && x < imgW)
								{
									for (int t = -1; t <= 1; ++t)
									{
										if (k == 0 && t == 0)
											continue;
										int y = currentPixel[1] + t;
										if (y >= 0 && y < imgH)
										{
											pPixel = pData + (y * imgW + x);
											if (*pPixel > 0)
											{
												currentComponent.push_back({ x, y });
												pixelQueue.push({ x, y });
												*pPixel = 0;
											}
										}
									}
								}
							}
						}
						components.push_back(currentComponent);
					}
				}
			}

			int maxId = -1;
			int maxSize = 0;
			for (int idx = 0; idx < components.size(); ++idx) {
				if (components[idx].size() > maxSize) {
					maxSize = components[idx].size();
					maxId = idx;
				}
			}

			// get result
			if (maxId >= 0) {
				int xMin = imgW + 1, xMax = -1, yMin = imgH + 1, yMax = -1;
				for (std::vector<int> pt : components[maxId]) {
					int k = pt[1] * imgW + pt[0];
					resData[k] = n;

					if (pt[0] > xMax) xMax = pt[0];
					if (pt[0] < xMin) xMin = pt[0];
					if (pt[1] > yMax) yMax = pt[1];
					if (pt[1] < yMin) yMin = pt[1];
				}

				// update box
				res.boxes.back() = { xMin, yMin, (xMax - xMin), (yMax - yMin) };
			}
		}
		
		n += 1;
	}
	
	//cv::imwrite("D:/srcImg.png", srcImg);

	return true;
}


void Yolact::Show(std::string imgPath, YolactDetectRes& res, const YolactDetectOption& opt)
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
		sprintf(text, "%s: %.2f", opt.classNames[res.classIds[idx]], res.scores[idx]);
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