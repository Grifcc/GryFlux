#pragma once

#include <string>

struct DetectBox
{
    DetectBox(float x1 = 0,
              float y1 = 0,
              float x2 = 0,
              float y2 = 0,
              float confidence = 0,
              float classID = -1,
              float trackID = -1)
        : x1(x1), y1(y1), x2(x2), y2(y2), confidence(confidence), classID(classID), trackID(trackID) {}

    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    float classID;
    float trackID;
};
