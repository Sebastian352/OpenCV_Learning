// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>

wchar_t* projectPath;

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	_wchdir(projectPath);

	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testNegativeImageFast()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// The fastest approach of accessing the pixels -> using pointers
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Accessing individual pixels in a RGB 24 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Defining pointers to each matrix (8 bits/pixels) of the individual components H, S, V 
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// Defining the pointer to the HSV image matrix (24 bits/pixel)
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;	// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	_wchdir(projectPath);

	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(100);  // waits 100ms and advances to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	_wchdir(projectPath);

	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snap the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

void changeGrayLevels()
{
	int changeValueBy=50;
	scanf("%d", &changeValueBy);
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				if (val + changeValueBy < 255) {
					dst.at<uchar>(i, j) = val+changeValueBy;
				}
				else {
					dst.at<uchar>(i, j) = 255;
				}
					
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}


void showColoredSquare() {
	int height = 256, width=256;
	Mat_<Vec3b> dst(height, width);


	for (int i = 0; i < height / 2; i++) {
		for (int j = 0; j < width / 2; j++) {
			dst(i, j) = Vec3b(255, 255, 255);
		}
	}

	for (int i = height / 2; i < height; i++) {
		for (int j = 0; j < width / 2; j++) {
			dst(i, j) = Vec3b(0, 255, 0);
		}
	}

	for (int i = 0; i < height / 2; i++) {
		for (int j = width/2; j < width; j++) {
			dst(i, j) = Vec3b(0, 0, 255);
		}
	}

	for (int i = height/2; i < height; i++) {
		for (int j = width/2; j < width; j++) {
			dst(i, j) = Vec3b(0, 255, 255);
		}
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < 5; j++) {
			dst(i, j) = Vec3b(0, 0, 0);
			dst(i, 255-j) = Vec3b(0, 0, 0);
		}
	}

	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < width; j++) {
			dst(i, j) = Vec3b(0, 0, 0);
			dst(255-i, j) = Vec3b(0, 0, 0);
		}
	}

	for (int i = height/2-1; i < height/2+2; i++) {
		for (int j = 0; j < width; j++) {
			dst(i, j) = Vec3b(0, 0, 0);
		}
	}

	for (int i = 0; i < height; i++) {
		for (int j = width/2-1; j < width/2+2; j++) {
			dst(i, j) = Vec3b(0, 0, 0);
		}
	}

	imshow("Square", dst);
	waitKey();
}

void splitChannels() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat RChannel = Mat(height, width, CV_8UC3), GChannel = Mat(height, width, CV_8UC3),BChannel = Mat(height, width, CV_8UC3);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				RChannel.at<Vec3b>(i, j) = Vec3b(0,0,src.at<Vec3b>(i,j)[2]);
				GChannel.at<Vec3b>(i, j) = Vec3b(0, src.at<Vec3b>(i, j)[1],0);
				BChannel.at<Vec3b>(i, j) = Vec3b(src.at<Vec3b>(i, j)[0],0,0);

			}
		}
		imshow("RChannel", RChannel);
		imshow("GChannel", GChannel);
		imshow("BChannel", BChannel);

		waitKey();
	}
}

void BGRtoGrayScale() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat grayScale = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				grayScale.at<uchar>(i, j) = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2]) / 3;
			}
		}
		imshow("GrayScale", grayScale);
		waitKey();
	}
}

void grayScaleToBlack() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat blackWhite = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) > 50)
					blackWhite.at<uchar>(i, j) = 255;
				else
					blackWhite.at<uchar>(i, j) = 0;
			}
		}

		imshow("BlackWhite", blackWhite);
		waitKey();
	}
}

void BGRtoHSV() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat H = Mat(height, width, CV_8UC1), S = Mat(height, width, CV_8UC1), V = Mat(height, width, CV_8UC1);
		
		
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				float r = float(src.at<Vec3b>(i, j)[2]) / 255;
				float g = float(src.at<Vec3b>(i, j)[1]) / 255;
				float b = float(src.at<Vec3b>(i, j)[0]) / 255;

				float M = max(r, max(b, g));
				float m = min(r, min(b, g));
				float C = M - m;

				float S_val;
				float V_val = M;
				float H_val;

				if (V_val != 0)
					S_val = C / V_val;
				else
					S_val = 0;

				if (C != 0) {
					if (M == r) H_val = 60 * (g - b) / C;
					if (M == g) H_val = 120 + 60 * (b - r) / C;
					if (M == b) H_val = 240 + 60 * (r - g) / C;
				}
				else
					H_val = 0;

				if (H_val < 0)
					H_val = H_val + 360;
				
				int H_norm = H_val * 255 / 360;
				int S_norm = S_val * 255;
				int V_norm = V_val * 255;

				H.at<uchar>(i, j) = H_norm;
				S.at<uchar>(i, j) = S_norm;
				V.at<uchar>(i, j) = V_norm;

			}
		}

		imshow("H value", H);
		imshow("S value", S);
		imshow("V value", V);

		waitKey();
	}
}



/* Histogram display function - display a histogram using bars (simlilar to L3 / Image Processing)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void histogramGenerator() {
	int h[256] = { 0 };
	float p[256];
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst;
		dst.create(src.size(), src.type());

		float M = width * height;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				h[src.at<uchar>(i, j)]++;
			}
		}
		for (int i = 0; i < 256; i++) {
			p[i] = h[i] / M;
		}

		int WH = 5;
		float TH = 0.0003;

		std::vector<int> maxh;
		maxh.push_back(0);
		for (int i = WH; i < 255 - WH + 1; i++) {
			float avg = 0;
			bool max = true;
			for (int k = i - WH; k < i + WH + 1; k++) {
				avg += p[k];
				if (p[k] > p[i]) {
					max = false;
					break;
				}
			}
			avg /= (2.0 * WH + 1);

			if (max && p[i] > avg + TH) {
				maxh.push_back(i);
			}
		}
		maxh.push_back(255);
		for (int i = 0; i < maxh.size(); i++) {
			printf("%d ", maxh[i]);
		}

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				for (int k = 1; k < maxh.size(); k++) {
					if (src.at<uchar>(i, j) > maxh[k - 1] && src.at<uchar>(i, j) < maxh[k]) {
						if (src.at<uchar>(i, j) - maxh[k - 1] < maxh[k] - src.at<uchar>(i, j)) {
							dst.at<uchar>(i, j) = maxh[k - 1];
						}
						else {
							dst.at<uchar>(i, j) = maxh[k];
						}
					}
				}
			}
		}

		imshow("Not Greyed", src);
		imshow("Greyed", dst);
		waitKey();
	}
}

int saturateInteger(int n) {
	if (n > 255)
		return 255;

	if (n < 0)
		return 0;

	return n;
}

void floydSteinberg() {
	int h[256] = { 0 };
	float p[256];
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(width, height, CV_8UC1);

		float M = width * height;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				h[src.at<uchar>(i, j)]++;
			}
		}
		for (int i = 0; i < 256; i++) {
			p[i] = h[i] / M;
		}

		int WH = 5;
		int maxh[256] = { 0 }, hcount = 1;
		float TH = 0.0003;

		for (int i = WH; i < 256 - WH; i++) {
			float avg = 0;
			bool max = true;
			for (int k = i - WH; k < i + WH + 1; k++) {
				avg += p[k];
				if (p[k] > p[i]) {
					max = false;
				}
			}
			avg /= 2.0 * WH + 1;

			if (max && p[i] > avg + TH) {
				maxh[hcount] = i;
				hcount++;
			}
		}
		maxh[hcount] = 255;
		hcount++;

		for (int i = 1; i < height-1; i++) {
			for (int j = 1; j < width-1; j++) {
				for (int k = 1; k < hcount; k++) {
					if (src.at<uchar>(i, j) > maxh[k - 1] && src.at<uchar>(i, j) < maxh[k]) {
						if (src.at<uchar>(i, j) - maxh[k - 1] < maxh[k] - src.at<uchar>(i, j)) {
							dst.at<uchar>(i, j) = maxh[k - 1];
						}
						else {
							dst.at<uchar>(i, j) = maxh[k];
						}
						int error = src.at<uchar>(i, j) - dst.at<uchar>(i, j);

						src.at<uchar>(i, j + 1) = saturateInteger(src.at<uchar>(i, j + 1) + 7 * error / 16);
						src.at<uchar>(i+1, j-1) = saturateInteger(src.at<uchar>(i + 1, j - 1) + 3 * error / 16);
						src.at<uchar>(i+1, j) = saturateInteger(src.at<uchar>(i + 1, j) + 5 * error / 16);
						src.at<uchar>(i+1, j + 1) = saturateInteger(src.at<uchar>(i + 1, j + 1) + error / 16);
					}
				}
			}
		}

		imshow("Not Greyed", src);
		imshow("Greyed", dst);
		waitKey();
	}
}




int main() 
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative\n");
		printf(" 4 - Image negative (fast)\n");
		printf(" 5 - BGR->Gray\n");
		printf(" 6 - BGR->Gray (fast, save result to disk) \n");
		printf(" 7 - BGR->HSV\n");
		printf(" 8 - Resize image\n");
		printf(" 9 - Canny edge detection\n");
		printf(" 10 - Edges in a video sequence\n");
		printf(" 11 - Snap frame from live video\n");
		printf(" 12 - Mouse callback demo\n");
		printf(" 13 - Change gray leves demo\n");
		printf(" 14 - Show colored square\n");
		printf(" 15 - Split image channels\n");
		printf(" 16 - BGRtoGrayScale\n");
		printf(" 17 - GrayScale to Black & White\n");
		printf(" 18 - BGR to HSV\n");
		printf(" 19 - Histogram\n");
		printf(" 20 - Floyd-Steinberg\n");

		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testNegativeImage();
				break;
			case 4:
				testNegativeImageFast();
				break;
			case 5:
				testColor2Gray();
				break;
			case 6:
				testImageOpenAndSave();
				break;
			case 7:
				testBGR2HSV();
				break;
			case 8:
				testResize();
				break;
			case 9:
				testCanny();
				break;
			case 10:
				testVideoSequence();
				break;
			case 11:
				testSnap();
				break;
			case 12:
				testMouseClick();
				break;
			case 13:
				changeGrayLevels();
				break;
			case 14:
				showColoredSquare();
				break;
			case 15:
				splitChannels();
				break;
			case 16:
				BGRtoGrayScale();
				break;
			case 17:
				grayScaleToBlack();
				break;
			case 18:
				BGRtoHSV();
				break;
			case 19:
				histogramGenerator();
				break;
			case 20:
				floydSteinberg();
				break;
		}
	}
	while (op!=0);

	return 0;
}