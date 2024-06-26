// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <fstream>
wchar_t* projectPath;

using namespace std;

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

void GeometricalFeaturesComputation(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDBLCLK)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
		Vec3b color = src->at<Vec3b>(y, x);
		int width = src->cols;
		int height = src->rows;
		int area = 0;
		float r=0, c=0, x=0,y=0;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src->at<Vec3b>(i, j) == color) {
					area++;

					r += i;
					c += j;
				}
			}
		}
		r /= area;
		c /= area;
		int NP = 0,cmin=INT_MAX,cmax=INT_MIN,rmin=INT_MAX,rmax=INT_MIN;

		for (int i = 1; i < height-1; i++) {
			for (int j = 1; j < width-1; j++) {
				if (src->at<Vec3b>(i, j) == color) {
					x += (i - r) * (j - c);
					y += (j - c) * (j - c) - (i - r) * (i - r);

					if (j < cmin)
						cmin = j;
					if (j > cmax)
						cmax = j;

					if (i < rmin)
						rmin = i;
					if (i > rmax)
						rmax = i;

					if (src->at<Vec3b>(i - 1, j - 1) != color || src->at<Vec3b>(i - 1, j) != color || src->at<Vec3b>(i - 1, j + 1) != color
						|| src->at<Vec3b>(i, j - 1) != color || src->at<Vec3b>(i, j + 1) != color || src->at<Vec3b>(i + 1, j - 1) != color
						|| src->at<Vec3b>(i + 1, j) != color || src->at<Vec3b>(i + 1, j + 1) != color) {
						NP++;
					}

				}
			}
		}
		x *= 2;
		float fi = atan2(x,y)/2,P;
		P = NP * CV_PI / 4;
		if (fi < 0) {
			fi = fi + CV_PI;
		}

		int ra = r + tan(fi)*(cmin-c);
		int rb = r + tan(fi) * (cmax - c);


		fi = fi * (180 / CV_PI);

		float thinnesRatio = 4 * CV_PI * area / (P * P);
		float aspectRatio= (float)(cmax - cmin + 1) / (float)(rmax - rmin + 1);


		//Projection 
		Mat dst = Mat(height, width, CV_8UC3,Scalar(255,255,255));


		for (int i = 0; i < height; i++) {
			int horz = 0;
			for (int j = 0; j < width; j++) {
				if (src->at<Vec3b>(i, j) == color) {
					dst.at<Vec3b>(i, horz) = color;
					horz++;
				}
			}
		}


		for (int i = 0; i < width; i++) {
			int vert = height - 1;
			for (int j = 0; j < height; j++) {
				if (src->at<Vec3b>(j, i) == color) {
					dst.at<Vec3b>(vert, i) = color;
					vert--;
				}
			}
		}

		imshow("Projection", dst);

		Point A(cmin, ra);
		Point B(cmax, rb);
		line(*src, A, B, Scalar(0, 0, 0), 2);

		imshow("El axis", *src);
		printf("area = %d r = %f c = %f FI = %f P = %f Thinnes = %f Aspect Ratio = %f \n", area,r,c,fi,P,thinnesRatio,aspectRatio);
		waitKey(0);
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

Mat grayScaleToBlack(Mat src) {
		int height = src.rows;
		int width = src.cols;
		Mat blackWhite = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) > 125)
					blackWhite.at<uchar>(i, j) = 255;
				else
					blackWhite.at<uchar>(i, j) = 0;
			}
		}

		//imshow("BlackWhite", blackWhite);
		//waitKey();
		return blackWhite;
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

vector<int> calcHist(Mat_<uchar> src) {

	vector<int> hist(256);
	int height = src.rows;
	int width = src.cols;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			hist[src(i,j)]++;
		}
	}
	return hist;
}

vector<float> calcNormHist(vector<int> hist,float M) {
	vector<float> norm(256);

	for (int i = 0; i < 256; i++) {
		norm[i] = hist[i] / M;
	}
	return norm;
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
		Mat dst = Mat(height,width,CV_8UC1);


		float M = width * height;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				h[src.at<uchar>(i, j)]++;
			}
		}
		for (int i = 0; i < 256; i++) {
			p[i] = h[i] / M;
		}

		showHistogram("Hist", h, 255, 500);

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
		Mat dst = src.clone();

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

						dst.at<uchar>(i, j + 1) = saturateInteger(dst.at<uchar>(i, j + 1) + 7 * error / 16);
						dst.at<uchar>(i+1, j-1) = saturateInteger(dst.at<uchar>(i + 1, j - 1) + 3 * error / 16);
						dst.at<uchar>(i+1, j) = saturateInteger(dst.at<uchar>(i + 1, j) + 5 * error / 16);
						dst.at<uchar>(i+1, j + 1) = saturateInteger(dst.at<uchar>(i + 1, j + 1) + error / 16);
					}
				}
			}
		}

		imshow("Not Greyed", src);
		imshow("Greyed", dst);
		waitKey();
	}
}

void geometricalFeatures() {
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", GeometricalFeaturesComputation, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

void labelBlackAndWhiteImageBFS() {
	Mat src,labels,dst;
	char fname[MAX_PATH];
	int di[8] = { 0,-1,-1,-1,0,1,1,1 };
	int dj[8] = { 1,1,0,-1,-1,-1,0,1 };

	while (openFileDlg(fname)) {
		src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int label = 0;
		labels = Mat(height, width, CV_32SC1,Scalar(0));
		dst = Mat(height, width, CV_8UC3);
		for (int i = 1; i < height - 1; i++) {
			for (int j = 1; j < width - 1; j++) {
				if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
					label++;

					Vec3b color = Vec3b(rand() % 256, rand() % 256, rand() % 256);
					std::queue<Point> Q;
					Q.push(Point(i,j));
					while (!Q.empty()) {
						Point q = Q.front();
						Q.pop();
						for (int k = 0; k < 8; k++) {
							if (src.at<uchar>(q.x + di[k], q.y + dj[k]) == 0 && labels.at<int>(q.x + di[k], q.y + dj[k]) == 0) {
								labels.at<int>(q.x + di[k], q.y + dj[k]) = label;
								Q.push(Point(q.x + di[k], q.y + dj[k]));
								dst.at<Vec3b>(q.x + di[k], q.y + dj[k]) = color;
							}
						}
					}
				}
			}
		}
		printf("%d\n", label);
		imshow("labeled", dst);
		waitKey(0);
	}
}


int minValueFromVector(std::vector<int> vec){
	int min = INT_MAX;
	for (int i = 0; i < vec.size(); i++) {
		if (min > vec[i]) {
			min = vec[i];
		}
	}
	return min;
}

void labelBlackAndWhiteImageTwoPass() {
	Mat src, labels, dst;
	char fname[MAX_PATH];
	int di[8] = { -1,-1,-1,0 };
	int dj[8] = { 1,0,-1,-1 };

	while (openFileDlg(fname)) {
		src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int label = 0;
		labels = Mat(height, width, CV_32SC1, Scalar(0));
		std::vector<std::vector<int>>edges(1000);

		dst = Mat(height, width, CV_8UC3);
		for (int i = 1; i < height - 1; i++) {
			for (int j = 1; j < width - 1; j++) {
				if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {

					std::vector<int> L;
					for (int k = 0; k < 4; k++) {
						if (labels.at<int>(i + di[k], j + dj[k]) > 0) {
							L.push_back(labels.at<int>(i + di[k], j + dj[k]));
						}
					}
					if (L.size() == 0) {
						label++;
						labels.at<int>(i, j) = label;
					}
					else {
						int x = minValueFromVector(L);

						labels.at<int>(i, j) = x;
						for (int k = 0; k < L.size(); k++) {
							if (L[k] != x) {
								edges[x].push_back(L[k]);
								edges[L[k]].push_back(x);
							}
						}

					}
				}
			}
		}

		int newLabel = 0;
		std::vector<int> newLabels(label + 1,0);

		for (int i = 0; i < label; i++) {
			if (newLabels[i] == 0) {
				newLabel++;
				std::queue<int> Q;
				newLabels[i] = newLabel;
				Q.push(i);
				while (!Q.empty()) {
					int x = Q.front();
					Q.pop();
					for (int k=0; k < edges[x].size(); k++) {
						if (newLabels[edges[x][k]] == 0) {
							newLabels[edges[x][k]] = newLabel;
							Q.push(edges[x][k]);
						}
					}
				}
			}
		}
		std::vector<Vec3b> colors;
		colors.push_back(Vec3b(0, 0, 0));
		for (int i = 1; i < newLabel+1; i++) {
			colors.push_back(Vec3b(rand() % 256, rand() % 256, rand() % 256));
		}

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				labels.at<int>(i, j) = newLabels[labels.at<int>(i, j)];
				dst.at<Vec3b>(i, j) = colors[labels.at<int>(i, j)];
			}
		}

		printf("%d\n", label);
		imshow("labeled", dst);
		waitKey(0);
	}
}

void borderAlgorithm() {

	Mat_<uchar> img = imread("./Images/triangle_up.bmp", IMREAD_GRAYSCALE);
	std::vector<int> dirs;
	std::vector<int> ddirs;
	std::vector<std::pair<int, int>> pts;
	int rows = img.rows;
	int cols = img.cols;
	Mat_<uchar> dst = Mat_<uchar>(rows, cols,150);

	int di[]= {0,-1,-1,-1,0,1,1,1};
	int dj[] = {1,1,0,-1,-1,-1,0,1};

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (img(i, j) == 0) {
				pts.push_back({i,j});
				goto nxt;
			}
		}
	}
nxt:
	int dir = 7;
	int n = 0;
	while (1) {
		n = n + 1;
		if (dir % 2 == 0) {
			dir = (dir + 7) % 8;
		}
		else {
			dir = (dir + 6) % 8;
		}
		for (int k = 0; k < 8; k++) {
			int dirnow = (dir + k) % 8;
			int i2 = pts.back().first + di[dirnow];
			int j2 = pts.back().second + dj[dirnow];
			if (img(i2, j2) == 0) {
				pts.push_back({ i2,j2 });
				dir = dirnow;
				dirs.push_back(dir);
				dst(i2, j2) = 0;
				break;
			}
		}
		if (n > 2 && pts[0] == pts[n - 1] && pts[1] == pts[n])
			break;
	}
	for (int i = 0; i < dirs.size()-1; i++) {
		ddirs.push_back((dirs[i + 1] - dirs[i] + 8) % 8);
		printf("%d %d\n", dirs[i],ddirs[i]);

	}

	imshow("imagine", dst);
	waitKey();
}

void reconstructImage() {

	std::vector<std::pair<int, int>> pts;
	int di[] = { 0,-1,-1,-1,0,1,1,1 };
	int dj[] = { 1,1,0,-1,-1,-1,0,1 };

	std::ifstream file("./Images/reconstruct.txt");
	if (!file.is_open()) {
		printf("error opening file");
	}

	int start_x, start_y;
	file >> start_x >> start_y;

	int vectorSize;
	file >> vectorSize;

	std::vector<int> vec(vectorSize);
	for (int i = 0; i < vectorSize; ++i) {
		file >> vec[i];
	}
	Mat_<uchar> img = imread("./Images/gray_background.bmp", IMREAD_GRAYSCALE);
	int n = 0;

	for (int i = 0; i < vectorSize; i++) {
		img(start_x, start_y) = 0;
		start_x += di[vec[i]];
		start_y += dj[vec[i]];
	}
	imshow("imag", img);
	waitKey();
}

Mat_<uchar> dilation(Mat_<uchar> img) {
	Mat_<uchar> strel(3, 3);
	strel.setTo(0);

	Mat_<uchar> dst = Mat_<uchar>(img.rows,img.cols,255);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == 0) {
				for (int u = 0; u < strel.rows; u++) {
					for (int v = 0; v < strel.cols; v++) {
						//vecinul care este sub elementul (u,v) din strel
						if (strel(u, v) == 0) {
							int i2 = i + u + strel.rows / 2;
							int j2 = j + v + strel.cols / 2;
							if (i2 >= 0 && i2 < img.rows && j2 < img.cols && j2 >= 0) {
								dst(i2, j2) = 0;
							}
						}
					}
				}
			}
		}
	}
	imshow("dilation", dst);
	waitKey();

	return dst;
}

bool insideImage(const Mat&img, int x, int y) {
	return x >= 0 && x < img.rows&& y >= 0 && y < img.cols;
}

Mat_ < uchar> erosion(Mat_<uchar> img) {
	Mat_<uchar> strel(3, 3);
	strel.setTo(0);

	Mat_<uchar> dst = Mat_<uchar>(img.rows, img.cols,255);

	for (int i = 1; i < img.rows-1; i++) {
		for (int j = 1; j < img.cols-1; j++) {
			if (img(i, j) == 0) {
				bool allblack = true;
				for (int u = 0; u < strel.rows; u++) {
					for (int v = 0; v < strel.cols; v++) {
						//vecinul care este sub elementul (u,v) din strel
						if (strel(u, v) == 0) {
							int i2 = i + u + strel.rows / 2;
							int j2 = j + v + strel.cols / 2;
							if (insideImage(img, i2, j2)) {
								if (img(i2, j2) == 255) {
									allblack = false;
								}
							}
						}
					}
				}
				if (allblack) {
					dst(i+strel.rows / 2, j+strel.cols / 2) = 0;
				}
				else {
					dst(i + strel.rows / 2, j + strel.cols / 2) = 255;
				}
			}
		}
	}
	imshow("erosion", dst);
	waitKey();
	return dst;
}

float meanValueOfIntensity(vector<float> p) {

	float mean = 0;

	for (int i = 0; i < 256; i++) {
		mean += i * p[i];

	}

	return mean;
}

float standardDev(vector<float>p) {

	float mean = meanValueOfIntensity(p);
	float dev = 0;

	for (int i = 0; i < 256; i++) {
		dev += (i - mean) * (i - mean) * p[i];
	}
	dev = sqrt(dev);

	return dev;
}

int cumulativeHist(vector<int>h,int bins) {
	int sum = 0;
	for (int i = 0; i < bins; i++) {
		sum += h[i];
	}
	return sum;
}

void basicGlobalThresh(Mat_<uchar> src,vector<int> h) {
	int Imin = 255, Imax = 0;
	for (int i = 0; i < 256; i++) {
		if (h[i] > 0) {
			Imax = i;
		}
		if (h[i] > 0 && i < Imin) {
			Imin = i;
		}
	}



	vector<float> T;

	T.push_back((Imin + Imax) / 2.0);
	float N1=0, N2=0;

	for (int i = Imin; i < T[0]; i++) {
		N1 += h[i];
	}

	for (int i = T[0] + 1; i < Imax; i++) {
		N2 += h[i];
	}
	float mew1, mew2;

	float g1=0, g2 = 0;


	for (int i = Imin; i < T[0]; i++) {
		g1 += i * h[i];
	}
	for (int i = T[0] + 1; i < Imax; i++) {
		g2 += i * h[i];
	}



	mew1 = g1 / N1;
	mew2 = g2 / N2;

	T.push_back((mew1 + mew2) / 2);

	float err = 0.1;

	int k = T.size()-1;

	printf("%f %f", T[k] - T[k - 1],T[k]);


	while (T[k] - T[k-1] > err){

		for (int i = Imin; i < T[k]; i++) {
			N1 += h[i];
		}
		for (int i = T[k] + 1; i < Imax; i++) {
			N2 += h[i];
		}

		for (int i = Imin; i < T[k]; i++) {
			g1 += i * h[i];
		}
		for (int i = T[k] + 1; i < Imax; i++) {
			g2 += i * h[i];
		}

		T.push_back((mew1 + mew2) / 2);
		k = T.size()-1;
	}

	int thresh = T[k];

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src(i, j) > thresh) {
				src(i, j) = 255;
			}
			else {
				src(i, j) = 0;
			}
		}
	}
	imshow("thresh",src);
	waitKey(0);
}

vector<int> imageNegative(vector<int> h) {
	vector<int> result(256);
	for (int i = 0; i < 256; i++) {
		result[i] = 255 - h[i];
	}

	return result;
}


vector<int> histSlide(vector<int> h,int offset) {
	vector<int> result(256);
	for (int i = 0; i < 256; i++) {
		if (h[i] + offset >= 0 && h[i] + offset <= 256) {
			result[i] = h[i + offset] ;
		}
	}

	return result;
}
vector<int> histStretchShrink(vector<int> h, int goutmax,int goutmin) {
	vector<int> result(256);
	int Imin=255, Imax=0;
	for (int i = 0; i < 256; i++) {
		if (h[i] > 0) {
			Imax = i;
		}
		if (h[i] > 0 && i < Imin) {
			Imin = i;
		}
	}

	float ratio = (float)(goutmax - goutmin) / (Imax - Imin);

	for (int i = 0; i < 256; i++) {
		result[i] = goutmax + (h[i] - Imin) * ratio;
	}

	int *arr = result.data();
	showHistogram("result", arr,255,200);
	waitKey(0);
	return result;
}

vector<int> gammaCorrection(vector<int> h, float gamma) {
	vector<int> result(256);
	for (int i = 0; i < 256; i++) {
		result[i] = 255 * pow(h[i] / 255.0, gamma);
		if (result[i] > 255) {
			result[i] = 255;
		}
		else if(result[i] < 0) {
			result[i] = 0;
		}
	}
	return result;
}

Mat_<float> conv(const Mat_<uchar>& src, const Mat_<float>& H) {
	int height = src.rows;
	int width = src.cols;
	int h = H.rows;
	int w = H.cols;
	int h_half = h / 2;
	int w_half = w / 2;

	// Initialize the result matrix with zeros
	Mat_<float> res(height, width, 0.0f);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float sum = 0.0f;
			for (int u = 0; u < h; u++) {
				for (int v = 0; v < w; v++) {
					int i2 = i + u - h_half;
					int j2 = j + v - w_half;
					if (insideImage(src, i2, j2)) {
						sum += H(u, v) * src(i2, j2);
					}
				}
			}
			res(i, j) = sum;
		}
	}

	return res;
}

Mat_<uchar> norm(Mat_<float>dst, Mat_<float>H) {
	//calculati cat este a si b din H
	//aplicati rescalare pe fiecare valoare din dst si salvati in img din iesire
	//daca nu exista nr negative a=0,b=sum(h)*255
	//daca exista nr negative a = suma elementelor negative *255,b = suma elementelor posizitive *255
	Mat_<uchar> res(dst.rows, dst.cols);
	int a = 0;
	int b = 0;
	int c = 0;
	int d = 255;
	int mx = 0;
	for (int i = 0; i < H.rows; i++) {
		for (int j = 0; j < H.cols; j++) {
			if (H(i, j) < 0) {
				a += H(i, j);
			}
			else {
				b += H(i, j);
			}
		}
	}

	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			dst(i, j) = abs(dst(i, j));
			mx = max(mx, dst(i, j));
		}
	}
	a *= mx;
	b *= mx;
	
	
	printf("%d %d %d %d\n", a, b, c, d);
	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			res(i, j) = (dst(i, j) - a) * (d - c) / (b - a) + c;
		}
	}
	return res;
}

Mat_ <uchar> norm2(Mat_<float>dst, Mat_<float>H) {
	Mat_<uchar> res(dst.rows, dst.cols);
	int a = 0;
	int b = 0;
	int c = 0;
	int d = 255;
	for (int i = 0; i < H.rows; i++) {
		for (int j = 0; j < H.cols; j++) {
			if (H(i, j) < 0) {
				a += H(i, j);
			}
			else {
				b += H(i, j);
			}
		}
	}
	a *= 255;
	b *= 255;
	// x in [-sumneg*255,sumpos*255]
	// abs(x) in [0,max(abs(sumneg),sumpos)*255]
	//

	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			res(i, j) = (abs(dst(i, j)) - a) * (d - c) / (b - a) + c;
		}
	}


	return res;
}

Mat_<float> kernel(int hv,int w) {
	Mat_<float> H = Mat_<float>(1, 2*w+1);
	int cp = 1;
	int cp2 = -w-1;
	for (int i = 0; i < 2*w+1; i++) {
		
		if (cp <= w) {
			H(0, i) = cp;
			cp++;
		}
		else {
			H(0, i) = cp2;
			cp2++;
		}
	}
	H(w / 2 + 2) = 0;



	return H;
}

Mat_<float> kernel2d(int w) {
	return Mat::zeros(w, w, CV_8UC1);
}

Mat_ <uchar> norm3(Mat_<float>dst, Mat_<float>H) {
	// min -> 255
	// max -> 0
	// 0 ->128
	
	// x [a,b]  y [c,d]

	//y  = (x - a) * (d - c) / (b - a) + c;

	Mat_<uchar> res(dst.rows, dst.cols);
	int a = 0;
	int b = 0;
	int c = 255;
	int d = 0;

	for (int i = 0; i < H.rows; i++) {
		for (int j = 0; j < H.cols; j++) {
			if (H(i,j) < 0) {
				a += H(i, j);
			}
			else {
				b += H(i, j);
			}
		}
	}
	a *= 255;
	b *= 255;

	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			res(i, j) = (dst(i, j) - a) * (d - c) / (b - a) + c;
		}
	}
	return res;
}

Mat_ <uchar> detectEdge(Mat_<float>con, Mat_<float> H) {
	int sum = 0;
	int t = 128;
	int height = con.rows;
	int width = con.cols;
	
	Mat_<uchar> res = Mat::zeros(height,width,CV_8UC1);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (abs(con(i,j)) > t) {
				res(i, j) = 255;
			}
			else {
				res(i, j) = 0;
			}

		}
	}
	return res;
}

Mat_<uchar> median_filter(Mat_<uchar> img, int w) {
	//fereastra wxw
	// sortam vecinii
	//pixelul din mijloc devine mediana
	Mat_<uchar> res(img.rows,img.cols);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			vector<int> filter;
			for (int k = i - w/2; k <= i + w/2; k++) {
				for (int l = j - w / 2; l <= j + w / 2; l++) {
					if (insideImage(img, k, l) == false) {
						continue;
					}
					filter.push_back(img(k, l));
				}
			}
			sort(filter.begin(), filter.end());
			res(i, j) = filter[filter.size() / 2];
		}
	}
	return res;
}

Mat_<float> createGaussianKernel(int w) {
	Mat_<float> kernel(w, w);
	int center = w / 2;
	float sum = 0.0;
	float sigma = w / 6.0;

	for (int i = 0; i < w; i++) {
		for (int j = 0; j < w; j++) {
			float x = i - center;
			float y = j - center;
			kernel(i, j) = (1 / (2 * PI * sigma * sigma)) * exp(-(x * x + y * y) / (2 * sigma * sigma));
			sum += kernel(i, j);
		}
	}

	// Normalize the kernel
	kernel /= sum;

	return kernel;
}


Mat_<uchar> Gauss2d(const Mat_<uchar>& img, int w) {
	float sigma = w / 6.0;
	Mat_<float> kernel = createGaussianKernel(w);

	for (int i = 0; i < kernel.rows; i++) {
		for (int j = 0; j < kernel.cols; j++) {
			printf("%f ", kernel(i, j));
		}
		printf("\n");
	}

	// Convert back to uchar for displaying
	Mat_<float> filteredFloatImg = conv(img, kernel);


	// Convert back to uchar for displaying
	Mat_<uchar> resultImg;
	filteredFloatImg.convertTo(resultImg, CV_8UC1);
	return resultImg;
}

void createGaussian1DKernel(int w, Mat_<float> &kernelX, Mat_<float> &kernelY) {

	int center = w / 2;
	float sigma = w / 6.0;

	for (int i = 0; i < w; i++) {
		float x = i - center;
		kernelX(0,i) = (1 / (sqrt((2 * PI)) * sigma) * exp(-(x * x) / (2 * sigma * sigma)));
	}


	for (int j = 0; j < w; j++) {
		float y = j - center;
		kernelY(j,0) = (1 / (sqrt(2 * PI)*sigma) * exp(-(y * y) / (2 * sigma * sigma)));
	}
}


Mat_<uchar> Gauss1d_1d(Mat_<uchar> img, int w) {
	//aplicam 10.8
	//filtram imaginea cu Gy, apoi rezultatul cu Gx
	//dst = Gx o (Gy o img)
	Mat_<float> kernelX(1, w);
	Mat_<float> kernelY(w, 1);
	createGaussian1DKernel(w,kernelX, kernelY);

	auto conv1 = conv(img, kernelY);
	auto conv2 = conv(conv1, kernelX);

	Mat_<uchar> resultImg;
	conv2.convertTo(resultImg, CV_8U);
	return resultImg;
}



/*
* 1. Create sobel kernels Sx and Sy (float)
* 
* 2. Calculate derivative in x and y direction
* 
* dx = src o Sx
* dy = src o Sy
* 
* floating point images, without normalization
* -for visualization imshow("dx", abs(dx)/255)
* 
* 3. Calculate magnitude and angle
* 
*  mag = sqrt(dx*dx + dy*dy)
*  angle = atan2(dy,dx) [-pi,pi]
*  - for visualization imshow( "Mag" ,abs(mag)/255)
*  4 non-maximum suppression
*  quantize angles q = ((int)round(angle/(2pi)*8)) % 8
*  if mag(i,j) > than neighbor its neighbor in direction and (q+4)%8 
*  mag2(i,j) = mag(i,j)
*  otherwise erase mag2(i,j) = 0
*  - for visualization imshow( "Mag2" ,abs(mag2)/255)
*/
void sobelKernel(Mat& Sx, Mat& Sy) {
	Sx = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);

	Sy = (Mat_<float>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
}

Mat_<uchar> edgeExtension(static Mat_<uchar> ext){
	Mat_<uchar> dst = ext.clone();
	int di[8] = { 0,-1,-1,-1,0,1,1,1 };
	int dj[8] = { 1,1,0,-1,-1,-1,0,1 };

	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			if (dst(i, j) == 255) {
				std::queue<Point> Q;
				Q.push(Point(i, j));

				while (!Q.empty()) {
					Point q = Q.front();
					Q.pop();
					for (int k = 0; k < 8; k++) {
						if (insideImage(dst, q.x + di[k], q.y + dj[k])) {
							if (dst(q.x + di[k], q.y + dj[k]) == 128) {
								dst(q.x + di[k], q.y + dj[k]) = 255;
								Q.push(Point(q.x + di[k], q.y + dj[k]));
							}
						}
					}
				}
			}
			else {
				dst(i, j) = 0;
			}
		}
	}
	imshow("edge extension", dst);
	waitKey(0);
	return dst;
}


Mat_<uchar> classifyEdges(Mat_<float> mag2) {
	Mat_<uchar> dst(mag2.rows, mag2.cols);
	float thigh = 120;
	float tlow = 80;

	for (int i = 0; i < mag2.rows; i++) {
		for (int j = 0; j < mag2.cols; j++) {
			if (mag2(i, j) > thigh) {
				dst(i, j) = 255;
			}
			else if (mag2(i, j) < tlow) {
				dst(i, j) = 0;
			}
			else {
				dst(i, j) = 128;
			}
		}
	}
	imshow("edge classification", dst);

	edgeExtension(dst);
	return dst;
}

Mat_<float> maximumSupression(Mat_<float> angle, Mat_<float> mag) {
	Mat_<float> mag2(mag.rows,mag.cols);
	int di[8] = { 0,-1,-1,-1,0,1,1,1 };
	int dj[8] = { 1,1,0,-1,-1,-1,0,1 };

	for (int i = 0; i < mag2.rows; i++) {
		for (int j = 0; j < mag2.cols; j++) {
			if (angle(i, j) < 0) {
				angle(i, j) += 2 * PI;
			}
			int q = ((int)round(angle(i,j) / (2 * PI) * 8)) % 8;
			float neighbor1 = 0.0f, neighbor2 = 0.0f;
			
			// Get the coordinates of the two neighboring pixels along the gradient direction
			int ni1 = i + di[q];
			int nj1 = j + dj[q];
			int ni2 = i - di[q];
			int nj2 = j - dj[q];

			if (insideImage(mag, ni1, nj1) && insideImage(mag, ni2, nj2)) {
				neighbor1 = mag(ni1, nj1);
				neighbor2 = mag(ni2, nj2);

				if (mag(i, j) > neighbor1 && mag(i, j) > neighbor2) {
					mag2(i, j) = mag(i, j);
				}
				else {
					mag2(i, j) = 0;
				}
			}
			else {
				mag2(i, j) = 0;
			}
		}
	}
	imshow("mag2", abs(mag2) / 255);
	classifyEdges(mag2);
	return mag2;
}


Mat_<float> magnitude(Mat_<float> dx, Mat_<float>dy) {
	Mat_<float> mag(dx.rows, dx.cols);
	Mat_<float> angle(dx.rows, dx.cols);

	for (int i = 0; i < mag.rows; i++) {
		for (int j = 0; j < mag.cols; j++) {
			mag(i, j) = sqrt(dx(i, j) * dx(i, j) + dy(i, j) * dy(i, j));
			angle(i, j) = atan2(dy(i, j), dx(i, j));
		}
	}

	maximumSupression(angle, mag);
	return mag;
}

Mat_<float> imageGradient(Mat_<uchar>src) {

	Mat Sx, Sy;
	sobelKernel(Sx, Sy);
	auto dx = conv(src, Sx);
	auto dy = conv(src, Sy);
	magnitude(dx, dy);
	return Sx;
}



int main() 
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);
    projectPath = _wgetcwd(0, 0);

	Mat_<uchar> src = imread("./Images/saturn.bmp", IMREAD_GRAYSCALE);
	imageGradient(src);

	/*
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
		printf(" 21 - GeometricalFeatures\n");
		printf(" 22 - Label Black & White image using BFS\n");
		printf(" 23 - Label Black & White image using Two-Pass\n");
		printf(" 24 - Border Algorithm \n");
		printf(" 25 - Reconstruct image \n");
		printf(" 26 - dilation \n");
		printf(" 27 - erosion \n");

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
			case 21:
				geometricalFeatures();
				break;
			case 22:
				labelBlackAndWhiteImageBFS();
				break;
			case 23:
				labelBlackAndWhiteImageTwoPass();
				break;
			case 24:
				borderAlgorithm();
				break;
			case 25:
				reconstructImage();
				break;
			case 26:
				dilation();
				break;
			case 27:
				erosion();
				break;
		}
	}
	while (op!=0);
	*/
	return 0;
}