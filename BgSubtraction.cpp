//
// This is based on Domenico Daniele Bloisi's tutorial
// available on http://docs.opencv.org/3.2.0/d1/dc5/tutorial_background_subtraction.html
//

#include "BgSubtraction.h"

using namespace cv;
using namespace std;
// Global variables
Mat frame; //current frame
Mat fgMaskMOG2; //fg mask fg mask generated by MOG2 method
Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
char keyboard; //input from keyboard

void processVideo(char* videoFilename);


int bgSub()
{
	//create GUI windows
	namedWindow("Frame");
	namedWindow("FG Mask MOG 2");
	//create Background Subtractor objects
	pMOG2 = createBackgroundSubtractorMOG2(); //MOG2 approach
	processVideo("Input.mp4");
	//destroy GUI windows
	destroyAllWindows();
	return EXIT_SUCCESS;
}
void processVideo(char* videoFilename) {
	//create the capture object
	VideoCapture capture(videoFilename);
	if (!capture.isOpened()) {
		//error in opening the video input
		cerr << "Unable to open video file: " << videoFilename << endl;
		exit(EXIT_FAILURE);
	}
	//read input data. ESC or 'q' for quitting
	keyboard = 0;
	while (keyboard != 'q' && keyboard != 27) {
		//read the current frame
		if (!capture.read(frame)) {
			cerr << "Unable to read next frame." << endl;
			cerr << "Exiting..." << endl;
			exit(EXIT_FAILURE);
		}
		//update the background model
		pMOG2->apply(frame, fgMaskMOG2);
		//get the frame number and write it on the current frame
		stringstream ss;
		rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
			cv::Scalar(255, 255, 255), -1);
		ss << capture.get(CAP_PROP_POS_FRAMES);
		string frameNumberString = ss.str();
		putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
			FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		//show the current frame and the fg masks
		imshow("Frame", frame);
		imshow("FG Mask MOG 2", fgMaskMOG2);
		//get the input from the keyboard
		keyboard = (char)waitKey(30);
	}
	//delete capture object
	capture.release();
}