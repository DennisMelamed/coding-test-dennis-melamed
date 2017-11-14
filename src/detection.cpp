#include "detection.h"

/*
Overall, the program flow will involve:
   Getting the the detection boxes from the input files
   Clustering those boxes (using kmeans)
   Pulling the most confident box out of each cluster
   Writing our results out and placing the most confident boxes over each image
*/


int main()
{
   int max_number_of_cars_per_photo = 10; //would need to vary based on how many cars could be expected
   int num_files = 5;                     //number of photos to parse
   mkdir("../solutions", 0777);           //this is a pretty unix-specific

   //cycles through the files
   for(int j = 0; j< num_files; j++)
   {
      //sets up some of the files we will need
      stringstream input_file, output_file, input_image, output_image;
      input_file << "../input/" << j;
      output_file << "../solutions/" << j;
      input_image << "../img/" << j << ".png";
      output_image << "../solutions/" << j << ".png";

      //gives us a list of Box structs we can work with
      vector<Box> boxes = readFile(input_file.str().c_str());

      //gives us the labels for the boxes data and the number of clusters
      BestFitLabelling labeledData = cluster(boxes, max_number_of_cars_per_photo);

      //gets the boxes with the most confidence out of each cluster
      vector<Box> bestBoxes = bestBoxFind(boxes, labeledData.labels, labeledData.k);

      //handles writing the output file
      writeResultsToFile(bestBoxes, output_file.str().c_str());

      //displays the image for us, also calls generateAndSaveImage which handles saving off the image with the selected boxes
      imshow(input_image.str().c_str(),generateAndSaveImage(bestBoxes, input_image.str().c_str(), output_image.str().c_str()));

   }
   
   waitKey(0);

   return 0;
}



/*
   cluster takes in a vector of detection boxes, and an int specifying the maximum number of clusters (i.e. vehicles) that could be in among the boxes
           It returns a struct containing the labels for the input vector of boxes, as well as how many clusters it decided were best in this case

           This method contains the meat of the algorithm, as it decides which groups of boxes really belong to the same vehicle.
           It uses kmeans to cluster the boxes based off of their top left point. A better implementation might use the center of each detection box,
           as this would probably more closely land on the vehicle.

           A simple heuristic was used to figure out how many clusters should be used:
           The compactness score for k-means is returned from the k-means method for each trial with k clusters. The differential between this
           score and the score for the next trial with k+1 clusters is computed. If this differential is dramatically greater than the next differential, 
           the value of k that is common to both differentials is probably the "elbow" of the graph of scores vs # of clusters. This value of k is used.
           If the no differential is greater than the previous by the factor chosen (100), then the same procedure is repeated with a new scaling factor (10),
           and the greatest value of k that meets the "elbow" criteria is chosen.

           A better heuristic might use a more mathematical approach to determining the best number of clusters, such as X-means, that depends less on the
           specific data set like this approach does.

*/
BestFitLabelling cluster(vector<Box> boxes, int max_clusters_to_try)
{

   Mat points(boxes.size(), 1, CV_32FC2);
   Mat labels;
   Mat centers;

   //transfer the top left points of each box to an openCV Mat
   for( int i = 0; i< boxes.size(); i++)
   {
      points.at<Vec2f>(i, 0)[0] = (float) boxes.at(i).x;
      points.at<Vec2f>(i, 0)[1] = (float) boxes.at(i).y;
   }


   //stop each call to kmeans after 10 iterations or when we converge closely enough
   TermCriteria crit(TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0);

   //determine the compactness score for each number of clusters
   double compactnesses[max_clusters_to_try+1];
   for(int i = 1; i<max_clusters_to_try+1; i++)
   {
      double compactness = kmeans(points, i, labels, crit, 3, KMEANS_PP_CENTERS, centers);
      compactnesses[i] = compactness;

   }


   //heuristic to determine best number of clusters based on differentials of compactnesses
   //this part could be improved dramatically with more time
   int best_k = 1;
   double prev_diff = 0;

   //if there is a large (i.e. more then 100 times) jump in the differential,
   //that is our elbow
   for( int i=2; i<max_clusters_to_try-1;i++)
   {
      double diff = (compactnesses[i-1] - compactnesses[i]);
      if(prev_diff > 100*diff)
      {
         best_k = i-1;
         kmeans(points, best_k, labels, crit, 3, KMEANS_PP_CENTERS, centers);
         BestFitLabelling result = {labels, best_k};
         return result;
      }
      else
      {
         prev_diff = diff;
      }

   }
   //there wasn't a huge jump, so take the max number of clusters that 
   //is located at a reasonably large change in compactness differential (>10 times)
   prev_diff = 0;
   for( int i=2; i<max_clusters_to_try-1;i++)
   {

      double diff = (compactnesses[i-1] - compactnesses[i]);
      if(prev_diff > 10*diff)
      {
         best_k = i-1;
         kmeans(points, best_k, labels, crit, 3, KMEANS_PP_CENTERS, centers);
      }
      else
      {
         prev_diff = diff;
      }

   }
   BestFitLabelling result = {labels, best_k};
   return result;
}


/*
   readFile opens the specified input file of detection boxes and outputs a vector of Box structs, which contain all the needed information
            about each box
*/
vector<Box> readFile(const char* path)
{
   stringstream ss;
   std::vector<Box> boxes;
   ifstream in(path);

   if(!in) {
      cout << "Cannot open input file.\n";
      cout << path;
   }

   char str[255];
   while(in) 
   {
     in.getline(str, 255);  
     if(in) 
     {
         ss << str << endl;
         float confidence;
         int x, y, width, height;

         if(in)ss >> confidence >> x >> y >> width >> height;

         Box current = {confidence, x, y, width, height};
         boxes.push_back(current);

      }
   }

   in.close();
   return boxes;
}


/*
   bestBoxFind takes in a vector of boxes, the labels for those boxes, and the number of clusters (i.e. classes) in those labels.
               It returns a vector of the box in each class that has the highest confidence score.
*/
vector<Box> bestBoxFind(vector<Box> boxes, Mat labels, int k)
{
   vector<Box> bestBoxes(k);
   for(int i = 0; i<k; i++)
   {
      bestBoxes[i] = {0,0,0,0,0};
   }
   for(int i = 0; i<boxes.size(); i++)
   {
      if(bestBoxes[labels.at<int>(i)].confidence < boxes[i].confidence)
      {
         bestBoxes[labels.at<int>(i)] = boxes[i];
      }
   }

   return bestBoxes;
}


/*
   writeResultsToFile takes in the boxes we wish to output as the ones defining a car, and where to save them as a path.
                      It writes them out to the specified location as x y width height for each box on a seperate line
*/
void writeResultsToFile(vector<Box> bestBoxes, const char* path)
{
   ofstream ofs(path, std::ofstream::out);

   for(Box box:bestBoxes)
   {
      ofs << box.x<< " " << box.y << " " << box.width << " " << box.height << "\n";
   }

   ofs.close();
}


/*
   generateAndSaveImage takes in the boxes we wish to draw on our image, the input image as a path, and
                        where to store the output image as a path.
                        It builds an image with the boxes and saves it as appropriate
*/
Mat generateAndSaveImage(vector<Box> bestBoxes, const char* input_image, const char* output_image)
{
   Mat img = imread(input_image,CV_LOAD_IMAGE_COLOR);
   for(int i=0; i<bestBoxes.size(); i++)
   {
      rectangle(img, Point(bestBoxes[i].x, bestBoxes[i].y), Point(bestBoxes[i].x+bestBoxes[i].width, bestBoxes[i].y+bestBoxes[i].height), Scalar( 0, 255, 255 ));
   }
   imwrite(output_image, img);
   return img;
}
