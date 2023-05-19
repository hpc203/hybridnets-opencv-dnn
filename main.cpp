#define _CRT_SECURE_NO_WARNINGS
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	string modelpath;
	string anchorpath;
};

class HybridNets
{
public:
	HybridNets(Net_config config);
	Mat detect(Mat frame);  ///opencv4.6可以正常推理
	~HybridNets();  // 析构函数, 释放内存
private:
	int inpWidth;
	int inpHeight;
	vector<string> det_class_names = { "car" };
	vector<string> seg_class_names = { "Background", "Lane", "Line" };
	int det_num_class;
	int seg_numclass;

	float confThreshold;
	float nmsThreshold;
	Net net;
	float* anchors = nullptr;
	const float mean_[3] = { 0.485, 0.456, 0.406 };
	const float std_[3] = { 0.229, 0.224, 0.225 };
	const bool keep_ratio = true;
	Mat resize_image(Mat srcimg, int *newh, int *neww, int *padh, int *padw);
	Mat normalize_(Mat img);
	vector<Vec3b> class_colors = { Vec3b(0,0,0), Vec3b(0, 255, 0), Vec3b(255, 0, 0) };
};

HybridNets::HybridNets(Net_config config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;

	this->net = readNet(config.modelpath);  ///opencv4.5在这里运行时异常了
	this->det_num_class = det_class_names.size();
	this->seg_numclass = seg_class_names.size();

	size_t pos = config.modelpath.rfind("_");
	size_t pos_ = config.modelpath.rfind(".");
	int len = pos_ - pos - 1;
	string hxw = config.modelpath.substr(pos + 1, len);

	pos = hxw.rfind("x");
	string h = hxw.substr(0, pos);
	len = hxw.length() - pos;
	string w = hxw.substr(pos + 1, len);
	this->inpHeight = stoi(h);
	this->inpWidth = stoi(w);

	pos = config.anchorpath.rfind("_");
	pos_ = config.anchorpath.rfind(".");
	len = pos_ - pos - 1;
	string len_ = config.anchorpath.substr(pos + 1, len);
	len = stoi(len_);
	this->anchors = new float[len];
	FILE* fp = fopen(config.anchorpath.c_str(), "rb");
	fread(this->anchors, sizeof(float), len, fp);//导入数据
	fclose(fp);//关闭文件。
}

HybridNets::~HybridNets()
{
	delete[] anchors;
	anchors = nullptr;
}

Mat HybridNets::resize_image(Mat srcimg, int *newh, int *neww, int *padh, int *padw)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_LINEAR);
			*padw = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *padw, this->inpWidth - *neww - *padw, BORDER_CONSTANT, 114);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_LINEAR);
			*padh = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *padh, this->inpHeight - *newh - *padh, 0, 0, BORDER_CONSTANT, 114);
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_LINEAR);
	}
	return dstimg;
}

Mat HybridNets::normalize_(Mat img)
{
	vector<cv::Mat> bgrChannels(3);
	split(img, bgrChannels);
	for (int c = 0; c < 3; c++)
	{
		bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1.0 / (255.0* std_[c]), (0.0 - mean_[c]) / std_[c]);
	}
	Mat m_normalized_mat;
	merge(bgrChannels, m_normalized_mat);
	return m_normalized_mat;
}

Mat HybridNets::detect(Mat srcimg)
{
	int newh = 0, neww = 0, padh = 0, padw = 0;
	Mat rgbimg;
	cvtColor(srcimg, rgbimg, COLOR_BGR2RGB);
	Mat dstimg = this->resize_image(rgbimg, &newh, &neww, &padh, &padw);
	Mat normalized_mat = this->normalize_(dstimg);
	
	
	Mat blob = blobFromImage(normalized_mat);
	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());   // 开始推理, opencv4.7在这里运行异常了

	float* classification = (float*)outs[0].data;
	float* box_regression = (float*)outs[1].data;
	float* seg = (float*)outs[2].data;

	vector<Rect> boxes;
	vector<float> confidences;
	vector<int> classIds;
	float ratioh = (float)srcimg.rows / newh, ratiow = (float)srcimg.cols / neww;
	const int num_proposal = outs[1].size[1];  ///输入的是单张图, 第0维batchsize忽略
	for (int n = 0; n < num_proposal; n++)
	{
		float conf = classification[n];
		/*int cls_id = -1;
		float max_conf = -10000;
		for (int k = 0; k < num_class; k++)   ////只有car这一个类，没必要求最大值
		{
			float conf = classification[n*num_class + k];
			if (conf > max_conf)
			{
				max_conf = conf;
				cls_id = k;
			}
		}*/

		if (conf > this->confThreshold)
		{
			const int row_ind = n * 4;
			float x_centers = box_regression[row_ind + 1] * this-> anchors[row_ind + 2] + this->anchors[row_ind];
			float y_centers = box_regression[row_ind] * this->anchors[row_ind + 3] + this->anchors[row_ind + 1];
			float w = exp(box_regression[row_ind + 3]) * this->anchors[row_ind + 2];
			float h = exp(box_regression[row_ind + 2]) * this->anchors[row_ind + 3];

			float xmin = (x_centers - w * 0.5 - padw)*ratiow;
			float ymin = (y_centers - h * 0.5 - padh)*ratioh;
			w *= ratiow;
			h *= ratioh;
			Rect box = Rect(int(xmin), int(ymin), int(w), int(h));
			boxes.push_back(box);
			confidences.push_back(conf);
			classIds.push_back(0);
		}
	}
	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);

	//// 画结果
	Mat outimg = srcimg.clone();
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		rectangle(outimg, Point(box.x, box.y), Point(box.x+box.width, box.y+box.height), Scalar(0, 0, 255), 2);
		string label = format("%.2f", confidences[idx]);
		label = this->det_class_names[classIds[idx]] + ":" + label;
		putText(outimg, label, Point(box.x, box.y - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 1);
	}

	const int area = this->inpHeight*this->inpWidth;
	int i = 0, j = 0, c = 0;
	for (i = 0; i < outimg.rows; i++)
	{
		for (j = 0; j < outimg.cols; j++)
		{
			const int x = int(j / ratiow) + padw;  ///从原图映射回到输出特征图
			const int y = int(i / ratioh) + padh;
			int max_id = -1;
			float max_conf = -10000;
			for (c = 0; c < seg_numclass; c++)
			{
				float seg_conf = seg[c*area + y * this->inpWidth + x];
				if (seg_conf > max_conf)
				{
					max_id = c;
					max_conf = seg_conf;
				}
			}
			if (max_id > 0)
			{
				outimg.at<Vec3b>(i, j)[0] = this->class_colors[max_id][0];
				outimg.at<Vec3b>(i, j)[1] = this->class_colors[max_id][1];
				outimg.at<Vec3b>(i, j)[2] = this->class_colors[max_id][2];
			}	
		}
	}
	return outimg;
}

int main()
{
	Net_config cfg = { 0.3, 0.5, "weights/hybridnets_256x384/hybridnets_256x384.onnx", "weights/hybridnets_256x384/anchors_73656.bin" }; 
	HybridNets net(cfg);
	string imgpath = "images/test.jpg";
	Mat srcimg = imread(imgpath);
	Mat outimg = net.detect(srcimg);

	static const string kWinName = "Deep learning object detection use OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, outimg);
	waitKey(0);
	destroyAllWindows();
}