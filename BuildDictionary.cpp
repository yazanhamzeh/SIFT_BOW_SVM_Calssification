#include <windows.h>
#include <stdio.h>
#include <iostream>
#include <conio.h> 
#include <fstream>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/ml.hpp"


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d; 
using namespace ml;

Ptr<SIFT > ptrSIFT = SIFT::create();
//create a nearest neighbor matcher using FLANN
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
//create BoF (or BoW) descriptor extractor
cv::BOWImgDescriptorExtractor bowDE(ptrSIFT, matcher);
//Declare SVM 
Ptr<ml::SVM> svm = ml::SVM::create();
std::vector<KeyPoint> trainKeypoints;
Mat featuresUnclustered, featuresClustered,dictionary;
Mat trainDescriptor;
int dictionarySize = 500;//Number of bags == size of Dictionary
Mat trainingData(0, dictionarySize, CV_32FC1);// to store training histogram
Mat labels(0, 1, CV_32FC1); // to store training sample class ID



//Writes results to a file
/**************************************************************************************************************/
//file reading/ writing function
//called from many functions 
//takes operation type read/write (OperationType),  file full path+name (FullFileName), file data label 
//(DataLabel) and (address of)data to be stored or loaded (data) path as arguments
//returns data stored to or loaded from file
/**************************************************************************************************************/
Mat ReadWriteFile(bool OperationType, string FullFileName,string DataLabel, Mat data)
{
	if (OperationType == 0)// read
	{
		FileStorage fs(FullFileName, FileStorage::READ);
		fs[DataLabel] >> data;
		fs.release();
		return data;
	}
	else //write
	{
		std::ofstream file1(FullFileName);
		FileStorage fs(FullFileName, FileStorage::WRITE);
		fs << DataLabel << data;
		fs.release();
		return data;
	}
}

/**************************************************************************************************************/
//Image file name reading function
//called from many functions 
//takes file names vector (filenames) and image directory(directory) path as arguments
//returns status 
/**************************************************************************************************************/
int readFilenames(std::vector<string> &filenames, const string &directory)
{
	HANDLE dir;
	WIN32_FIND_DATA file_data;

	if ((dir = FindFirstFile((directory + "/*").c_str(), &file_data)) == INVALID_HANDLE_VALUE)
		return -1; /* No files found */

	do {
		const string file_name = file_data.cFileName;
		const string full_file_name = directory + "//" + file_name;
		const bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
		
		

		if (is_directory)
			continue;

		filenames.push_back(full_file_name);
	} while (FindNextFile(dir, &file_data));

	FindClose(dir);
	return filenames.size();

}

/**************************************************************************************************************/
//SIFT Descriptor calculation function
//called from many functions 
//takes Image data (trainImg) as argument
//returns status 
/**************************************************************************************************************/

BOOL CalculateSIFT(cv::Mat trainImg)
{
	// Verify the images loaded successfully.
	if (trainImg.empty())
	{
		cerr<<"Can't read image\n" << endl;
		return false;
	}
	
	// Detect keypoints in the image.
	
	ptrSIFT->detect(trainImg, trainKeypoints);

	// Compute the SIFT feature descriptors for the keypoints.
	// Multiple features can be extracted from a single keypoint, so the result is a
	// matrix where row 'i' is the list of features for keypoint 'i'.
	
	ptrSIFT->compute(trainImg, trainKeypoints, trainDescriptor);
	

	return true;
	
}

/**************************************************************************************************************/
//BoW Dictionary generation function
//called from after "BuildDictionary" function
//takes full name of dictionary file to wrtie dictionary into as arguments
//returns nothing 
/**************************************************************************************************************/
void GenerateDictionary(string DB_File)
{
	//Construct BOWKMeansTrainer
	//define Term Criteria. 
	//Terminate after 100 iterations or as soon as cluster centers moves by less than 0.001
	TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
	//retries number. Run the algorithm only once.
	int retries = 1;
	//necessary flags
	//Use kmeans++ center initialization by Arthur and Vassilvitskii [Arthur2007].
	int flags = KMEANS_PP_CENTERS;
	//Create the BoW (or BoF) trainer
	BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
	//Report how many keypoints are to be lustered. For deugging only
		cout << "Clustering " << featuresUnclustered.rows << " features" << endl;
	//cluster the feature vectors
	dictionary = bowTrainer.cluster(featuresUnclustered);
	
	// Write dictionary to file
	Mat testing = ReadWriteFile(1, DB_File, "vocabulary", dictionary);
	
}

/**************************************************************************************************************/
//BoW clustering function
//Runs after "main" function
//takes SIFT clustering files(DB_File) and clustering Image folder (folder) as arguments
//returns status 
/**************************************************************************************************************/
int BuildDictionary(string folder, string DB_File)
{
	cout << "Reading in directory " << folder << endl;
	vector<string> filenames;

	int num_files = readFilenames(filenames, folder);
	cout << "Number of files = " << num_files << endl;// for debugging only
	cv::waitKey(-1);
	cv::namedWindow("image", 1); //create a window to show the loaded image 

	for (size_t i = 0; i < filenames.size(); ++i)
	{
		cout << folder + filenames[i] << " #" << i << endl; //for debugging only
		cv::Mat src = cv::imread(filenames[i], CV_LOAD_IMAGE_GRAYSCALE);
		if (src.data)
		{ //Protect against no file

			BOOL SIFT_st = CalculateSIFT(src);
			if (!SIFT_st)
			{
				cerr << "Error in SIFT process" << endl;
				return -1;
			}
			else
			{
				featuresUnclustered.push_back(trainDescriptor);
			}
		}
		else
		{
			cerr << folder + filenames[i] << ", file #" << i << ", is not an image" << endl;
			continue;
		}
		Mat trainImgOut;
		drawKeypoints(src, trainKeypoints, trainImgOut, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		cv::imshow("image", trainImgOut);
		waitKey(50);
	}
	// Create the Dictionary
	GenerateDictionary(DB_File);

	
	
}

/**************************************************************************************************************/
//BoW training function
//Runs after "GenerateDictionary" function
//takes BoW training files(Results_File,Label_File) and training Image folder (folder) as arguments
//returns status 
/**************************************************************************************************************/
int TrainUsingBoW(string folder, string DB_File, string Results_File, string Label_File)
{
		
	//To store the BoW (or BoF) representation of the training images
	
	//Load BOW descriptor_extractor from file    
	featuresClustered = ReadWriteFile(0, DB_File, "vocabulary", featuresClustered);
	/*FileStorage fs(DB_File, FileStorage::READ);
	fs["Histogram"] >> featuresClustered;
	fs.release();*/
	
	//Set the dictionary with the vocabulary we created in the first step
	bowDE.setVocabulary(featuresClustered);

	//Read Training image filenames from training directory
	cout << "Reading in directory " << folder << endl;
	vector<string> filenames;
	int num_files = readFilenames(filenames, folder);
	cout << "Number of files = " << num_files << endl;// for debugging only
		
	for (size_t i = 0; i < filenames.size(); ++i)
	{
		cout << folder + filenames[i] << " #" << i << endl; //for debugging only
		cv::Mat src = cv::imread(filenames[i], CV_LOAD_IMAGE_GRAYSCALE);
		if (src.data)
		{ //Protect against no file

		  // Get State ID from image filename
			
			BOOL SIFT_st = CalculateSIFT(src);
			if (!SIFT_st)
			{
				cerr << "Error in SIFT process" << endl;
				return -1;
			}
			else
			{
				//Display Training images
				Mat trainImgOut;
				drawKeypoints(src, trainKeypoints, trainImgOut, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
				cv::imshow("image", trainImgOut);
				waitKey(50);

				//extract BoW (or BoF) descriptor from given image
				//std::vector<std::vector<int> >* pointIdxsOfClusters = 0;
				Mat bowDescriptor;
					bowDE.compute( src, trainKeypoints, bowDescriptor);
					
				if (bowDescriptor.empty())
				{
						cerr << "Error in BOW process" << endl;
					
					return -1;
				}
				// Store visual codewords
				trainingData.push_back(bowDescriptor);
				int pos1 = filenames[i].find ("MI_");
				if(pos1==-1)
					labels.push_back(0);// 0== Not a Michigan Plate
				else
					labels.push_back(1); // 1== Michigan Plate
 
					
				
			}
		}
		else
		{
			cerr << folder + filenames[i] << ", file #" << i << ", is not an image" << endl;
			continue;
		}
		
	}
	//store the Visual CodeWords to file
	Mat testing1 = ReadWriteFile(1, Results_File, "CodeWords", trainingData);
		
	//store the class labels to file
	Mat testing2 = ReadWriteFile(1, Label_File, "ClassLabels", labels);
	
	return 0;

}

/**************************************************************************************************************/
//SVM training function
//Runs after "TrainUsingBoW" function
//Accepts BoW training files and SVM training outputfile paths as arguments
//returns status 
/**************************************************************************************************************/
bool TrainSVM(string SVM_DB_File,string TrainedResultsFile,string BoW_Labels_File)
{
	//Load training data and labels 
	Mat trainingDataMat = ReadWriteFile(0, TrainedResultsFile,  "CodeWords",  trainingData);
	Mat labelsMat = ReadWriteFile(0, BoW_Labels_File, "ClassLabels", labels);
	//Setting up SVM parameters
	
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::RBF);
	svm->setC(312.5);
	svm->setGamma(3);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-4));
	
	// Train the SVM with above parameters
	svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
	svm->save(SVM_DB_File);

	cout << "Processing evaluation data..." << endl;

	if (svm.empty())
		return 1;
	else
		return 0;
}

/**************************************************************************************************************/
//SVM prediction function
//called from main
//Accepts folder paths as arguments
//returns status 
/**************************************************************************************************************/
int Predict(Mat ImageData, string SVM_DB_File,  string DictionaryFile)
{
	
	//Load BOW descriptor_extractor from file 
	dictionary = ReadWriteFile(0, DictionaryFile, "vocabulary", featuresClustered);
	// Set the vocabulary
	bowDE.setVocabulary(dictionary);
	Mat descriptors;
	bowDE.compute(ImageData, trainKeypoints, descriptors);
	if (descriptors.empty())  
		return 0;

	// setup svm
	
	//svm->load(SVM_DB_File);
	Ptr<ml::SVM> svm = Algorithm::load<ml::SVM>(SVM_DB_File);
	//svm->save("C://OPENCV_3_2_0//OpenSource//Final//SIFT_BOW_SVM_DB//testfile.yml");
	
	float prediction=svm->predict(descriptors);
	cout << "Prdiction for this image is: " << prediction << endl;
	
	return (int)prediction;
}

/**************************************************************************************************************/
//SVN Predicting dat collection  function
//called from main
//Accepts folder paths as arguments
//returns status 
/**************************************************************************************************************/

bool CollectPredictionData(string TestImageFolder, string SVMFile, string DictionaryFile, string PredictionResultsFile)
{
	

	cout << "Reading in directory " << TestImageFolder << endl;
	vector<string> filenames;

	int num_files = readFilenames(filenames, TestImageFolder);
	//Mat Prediction_GrounTruth(3,filenames.size(), CV_32FC1);// to store SVM prediction results
	string Prediction_GrounTruth[3][256];
	cout << "Number of files = " << num_files << endl;// for debugging only
	cv::waitKey(-1);
	cv::namedWindow("image", 1); //create a window to show the loaded image 
	
	//Mat GrounTruth(0, 1, CV_32FC1);// to store actual test image classes

	ofstream fout(PredictionResultsFile);
	fout << " Sample#   Prediction    Actual\n";// labels
	for (size_t i = 0; i < filenames.size(); ++i)
	{
		cout << TestImageFolder + filenames[i] << " #" << i << endl; //for debugging only
		
		cv::Mat src = cv::imread(filenames[i], CV_LOAD_IMAGE_GRAYSCALE);
		if (src.data)
		{ //Protect against no file

			BOOL SIFT_st = CalculateSIFT(src);
			if (!SIFT_st)
			{
				cerr << "Error in SIFT process" << endl;
				return -1;
			}
			else
			{
				float SVM_Prediction = Predict(src, SVMFile, DictionaryFile);
				/*Prediction_GrounTruth.at<string>(0, i) = "Sample #" +to_string(i)+ "\t";
				Prediction_GrounTruth.at<string>(1,i) =to_string(SVM_Prediction)+"\t";*/
				Prediction_GrounTruth[0][i] = "\t"+ to_string(i) + "   \t";
				float x = SVM_Prediction;
				Prediction_GrounTruth[1][i] = "   "+ to_string((int)x) + "\t";
				int pos1 = filenames[i].find("MI_");
				if (pos1 == -1)
					//Prediction_GrounTruth.at<string>(2,i)=" 0\n";// 0== Not a Michigan Plate
					Prediction_GrounTruth[2][i] = "\t\t0\t\n";
				else
					//Prediction_GrounTruth.at<string>(2,i) =" 1\n"; // 1== Michigan Plate
					Prediction_GrounTruth[2][i] = "\t\t1\t\n";
				
				fout << Prediction_GrounTruth[0][i] +  Prediction_GrounTruth[1][i] + Prediction_GrounTruth[2][i];

			}
		}

		else
		{
			cerr << TestImageFolder + filenames[i] << ", file #" << i << ", is not an image" << endl;
			continue;
		}
		Mat trainImgOut;
		drawKeypoints(src, trainKeypoints, trainImgOut, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		cv::imshow("image", trainImgOut);
		waitKey(50);
	}
	//Write results to file
	//ReadWriteFile(1, PredictionResultsFile, "SVM Results", Prediction_GrounTruth);

	return 0;
}

/**************************************************************************************************************/
//Main function
//Runs first
//Accepts folder paths as arguments
//returns status 
/**************************************************************************************************************/
int main(int argc, char** argv)
{
	

	char ProcessType = 4;//1: Build dictionary, 2: BoW training, 3: Train SVM 4: Predict with SVM
	
		
	if (argc !=6) {
		cerr << "\nIncorrect number of parameters: " << argc << ", should be 2\n" << endl;
		return -1;
	}
	//Load directory structure
	string ClusteringFolder = argv[1];
	string TrainingFolder   = argv[2];
	string TestingFolder    = argv[3];
	string  DataBaseFolder  = argv[4];
	string  ResultsFolder   = argv[5];
	// assign file names
	//Results of BoW clustering 
	const string DictionaryFile = DataBaseFolder + "//" + "Dictionary.yml";
	//Results of training are stored two files
	const string TrainedResultsFile = DataBaseFolder + "//" + "TrainedHistogram.yml";
	const string TrainedLabelsFile = DataBaseFolder + "//" + "TrainedLabel.yml";
	const string SVM_DB_File = DataBaseFolder + "//TrainedSVM.yml";
	// Results of SVM classification
	const string SVM_ResultsFile = ResultsFolder + "//" + "ClassificationResults.yml";
	
	if(ProcessType==1)// Create Dictionary 
	{
		int status = BuildDictionary(ClusteringFolder, DictionaryFile);
		if (status == -1)
		{
			cerr << "Error in creating Dictionary" << endl;
			return -1;
		}
		else
		{
			return 0;
		}		

	}
	else 
		if(ProcessType == 2) // Bow training
		{
			int status = TrainUsingBoW(TrainingFolder, DictionaryFile, TrainedResultsFile, TrainedLabelsFile);
			if (status == -1)
			{
				cerr << "Error in BoW process" << endl;
				return -1;
			}
			else
			{
				return 0;
			}
			
		}
		else 
			if(ProcessType == 3)// train with SVM
			{
				
				bool SVMStatus= TrainSVM(SVM_DB_File, TrainedResultsFile, TrainedLabelsFile);
				if (SVMStatus)
				{
					cerr << "Error in SVM training" << endl;
					return -1;
				}
				return 0;
			}
			else//Predict with SVM
			{
				
				bool status = CollectPredictionData(TestingFolder, SVM_DB_File, DictionaryFile, SVM_ResultsFile);
					if(status)
					{
						cerr << "Error in Prediction process" << endl;
						return -1;
					}
						
				return 0;
			}
}


