//
// Created by cv on 2021/10/6.
//

#include "YOLOXMNN.h"
#include <vector>

YOLOXMNN::YOLOXMNN()
{

}
YOLOXMNN::~YOLOXMNN()
{

}
void  YOLOXMNN::GenGridBox(const int netWidth,
                            const int netHeight)
{
    for (int i = 0; i < 3; i++) {
        int gridRow = int(netHeight / mStrides[i]);
        int gridCol = int(netWidth / mStrides[i]);
        for(int row = 0; row < gridRow; row++)
        {
            for(int col = 0; col < gridCol; col++)
            {
                GridInfo gridInfo;
                gridInfo.gridX = col;
                gridInfo.gridY = row;
                gridInfo.stride = mStrides[i];
                mGridInfos.push_back(gridInfo);
            }
        }
    }
}
cv::Mat YOLOXMNN::PreprocImage(const cv::Mat& inputImage,
                               const int netWidth,
                               const int netHeight,
                               float& fRatio)
{
    int width = inputImage.cols, height = inputImage.rows;
    cv::Mat imageOut(netHeight, netWidth, CV_8UC3);
    if(width == netWidth && height == netHeight)
    {
        inputImage.copyTo(imageOut);
        return inputImage;
    }
    memset(imageOut.data, 114, netWidth * netHeight * 3);
    fRatio = std::min((float)netWidth /(float)width, (float)netHeight / (float)height);
    int newWidth = (int)(fRatio * width), newHeight = (int)(fRatio * height);
    cv::Mat rzImage;
    cv::resize(inputImage, rzImage, cv::Size(newWidth, newHeight));
    cv::Mat rectImage = imageOut(cv::Rect(0, 0, newWidth, newHeight));
    rzImage.copyTo(rectImage);
    return imageOut;

}
void YOLOXMNN::NMS(std::vector<DetBoxes>& detBoxes,
                   std::vector<int>& picked)
{
    std::sort(detBoxes.begin(), detBoxes.end(),
              [](const DetBoxes& a, const DetBoxes& b)
              {
                  return a.scoreObj > b.scoreObj;
              });
    picked.clear();
    const int n = (int)detBoxes.size();
    for (int i = 0; i < n; i++) {
        const DetBoxes &a = detBoxes[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const DetBoxes &b = detBoxes[picked[j]];
            // intersection over union
            float  x0 = std::max(a.x,  b.x);
            float  y0 = std::max(a.y, b.y);
            float  x1 = std::min(a.x + a.w, b.x + b.w);
            float  y1 = std::min(a.y + a.h, b.y + b.h);
            float inter_area = std::max(0.0f, (x1 - x0)) * std::max(0.0f , (y1 - y0));
            float union_area = a.area + b.area - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > mNMSThre)
                keep = 0;
        }
        if (keep)
            picked.push_back(i);
    }

}
bool  YOLOXMNN::loadWeight(const char* weightFile)
{
    mNet.reset(MNN::Interpreter::createFromFile(weightFile));
    if(mNet == nullptr)
    {
        return false;
    }
    mSession = mNet->createSession(mConfig);
    return true;
}
bool YOLOXMNN::Inference(const cv::Mat& inputImage)
{
    if(!mSession)
    {
        return false;
    }
    MNN::CV::Matrix trans;
    trans.setScale(1.0f, 1.0f);

    MNN::CV::ImageProcess::Config imageConfig;
    imageConfig.filterType = MNN::CV::BILINEAR;
    imageConfig.sourceFormat = MNN::CV::BGR;
    imageConfig.destFormat = MNN::CV::BGR;

    std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(imageConfig));
    pretreat->setMatrix(trans);

    MNN::Tensor* inputTensor = mNet->getSessionInput(mSession, NULL);
    std::vector<int> inputShape = inputTensor->shape();
    int netWidth = 0, netHeight = 0, netChannel = 0;
    netChannel = inputShape[1];
    netHeight = inputShape[2];
    netWidth = inputShape[3];
    MNN_PRINT("input: w:%d , h:%d, c: %d\n", netWidth, netHeight, netChannel);
    this->GenGridBox(netWidth, netHeight);
    MNN_PRINT("GRID SIZE: %d \n", (int)mGridInfos.size());

    float fRatio = 0;
    cv::Mat netImage = this->PreprocImage(inputImage, netWidth, netHeight, fRatio);

    pretreat->convert((uint8_t*)netImage.data, netImage.cols, netImage.rows, 0, inputTensor);
    mNet->runSession(mSession);

    MNN::Tensor* outputTensor = mNet->getSessionOutput(mSession, NULL);
    int outHW = 0, outChannel = 0;
    std::vector<int> outShape = outputTensor->shape();
    outHW = outShape[1];
    outChannel = outShape[2];
    MNN_PRINT("output: wh: %d, c: %d \n", outHW, outChannel);

    MNN::Tensor offset_host(outputTensor, outputTensor->getDimensionType());
    outputTensor->copyToHostTensor(&offset_host);
    float* outData = offset_host.host<float>();
    MNN_PRINT("outData: index:0 , value: %.2f \n", outData[0]);

    std::vector<DetBoxes> detBoxes;
    for (int i = 0; i < outHW; ++i, outData+=outChannel) {
        DetBoxes  detBox;
        float  centerX = (mGridInfos[i].gridX + outData[0]) * mGridInfos[i].stride;
        float  centerY = (mGridInfos[i].gridY + outData[1]) * mGridInfos[i].stride;
        detBox.w = std::exp(outData[2]) * mGridInfos[i].stride;
        detBox.h = std::exp(outData[3]) * mGridInfos[i].stride;
        detBox.x = centerX - detBox.w * 0.5f;
        detBox.y = centerY - detBox.h * 0.5f;
        detBox.iouScore = outData[4];
        float score = 0.0f;
        int clsIndex = 0;
        float* clsScoreData = outData + 5;
        for (int j = 0; j < (outChannel - 5); ++j) {
            if(score < clsScoreData[j])
            {
                score = clsScoreData[j];
                clsIndex = j;
            }
        }
        detBox.score = score;
        detBox.clsIndex = clsIndex;
        detBox.area = detBox.w * detBox.h;
        detBox.scoreObj = detBox.score * detBox.iouScore;
        if(detBox.scoreObj >= mClsThre) {
            detBoxes.push_back(detBox);
        }
    }
    std::vector<int> picked;
    this->NMS(detBoxes, picked);
    printf("det num: %d \n",(int) picked.size());
    for (int i = 0; i < (int) picked.size() ; ++i) {
        DetBoxes&  detPickedBox = detBoxes[picked[i]];
        cv::rectangle(netImage, cv::Point((int)detPickedBox.x,  (int)detPickedBox.y),
                                cv::Point((int)(detPickedBox.x + detPickedBox.w),  (int)detPickedBox.y + detPickedBox.h),
                                cv::Scalar(0, 0, 255), 2);
        printf("index: %d, score: %.2f \n", detPickedBox.clsIndex, detPickedBox.scoreObj);
    }
    cv::imshow("draw", netImage);
    cv::waitKey(-1);

    return true;
}
