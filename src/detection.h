#include "opencv2/core/core_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;

#include <sys/stat.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <math.h>
using namespace std;

//all the relevant info for each detection box
struct Box
{
   double confidence;
   int x;
   int y;
   int width;
   int height;

};

//a labelling and number of clusters for a set of boxes
struct BestFitLabelling
{
   Mat labels;
   int k;
};

BestFitLabelling cluster(vector<Box> boxes, int max_clusters_to_try);
vector<Box> readFile(const char* path);
vector<Box> bestBoxFind(vector<Box> boxes, Mat labels, int k);
void writeResultsToFile(vector<Box> bestBoxes, const char* path);
Mat generateAndSaveImage(vector<Box> bestBoxes, const char* input_image, const char* output_image);
