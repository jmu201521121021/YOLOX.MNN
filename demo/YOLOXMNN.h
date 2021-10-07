//
// Created by cv on 2021/10/6.
//

#ifndef YOLOX_MNN_YOLOXMNN_H
#define YOLOX_MNN_YOLOXMNN_H

#include<iostream>
#include<vector>
#include<string.h>

#include "opencv2/opencv.hpp"
#include <MNN/ImageProcess.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>

typedef  struct GridInfo
{
    float gridX;
    float gridY;
    float stride;
}GridInfo;

typedef struct DetBoxes
{
    float x;
    float y;
    float w;
    float h;
    float score;
    float iouScore;
    float area;
    float scoreObj;
    int clsIndex;
}DetBoxes;

class YOLOXMNN {
public:
    YOLOXMNN();
    ~YOLOXMNN();

    void  GenGridBox(const int netWidth, const int netHeight);
    cv::Mat PreprocImage(const cv::Mat& inputImage,
                         const int netWidth,
                         const int netHeight,
                         float& fRatio);
    void NMS(std::vector<DetBoxes>& detBoxes, std::vector<int>& picked);

    bool LoadWeight(const char* weightFile);
    bool Inference(const cv::Mat& inputImage, std::vector<DetBoxes>& detBoxes);
    void Postprocess(const MNN::Tensor* outTensor,
                     const float  ratio,
                     std::vector<DetBoxes>& outBoxes);

private:
    std::shared_ptr<MNN::Interpreter> mNet = nullptr;     // net
    MNN::Session*  mSession =  nullptr;                   // session
    MNN::ScheduleConfig mConfig;                          // session config, default
    MNN::CV::ImageProcess::Config mImageConfig;           // image config
    std::shared_ptr<MNN::CV::ImageProcess> mPretreat;     // image data to tensor

    int mNetWidth = 0, mNetHeight = 0, mNetChannel=0;
    MNN::Tensor* mInputTensor;
    float mStrides[3] = {8, 16, 32};
    std::vector<GridInfo> mGridInfos;
    float  mClsThre = 0.3f;
    float  mNMSThre = 0.3f;

};


#endif //YOLOX_MNN_YOLOXMNN_H
