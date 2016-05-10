#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;
/*
int main()
{
	Mat trainImage = imread("box.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat queryImage = imread("box_in_scene.png", CV_LOAD_IMAGE_GRAYSCALE);

	Ptr<Feature2D> detector = BRISK::create();

	vector<KeyPoint> trainKeypoints;
	Mat trainDescriptors;
	detector->detectAndCompute(trainImage, noArray(), trainKeypoints, trainDescriptors);

	vector<KeyPoint> queryKeypoints;
	Mat queryDescriptors;
	detector->detectAndCompute(queryImage, noArray(), queryKeypoints, queryDescriptors);

	BFMatcher matcher(NORM_HAMMING);

	vector<vector<DMatch>> initialMatches;
	matcher.knnMatch(queryDescriptors, trainDescriptors, initialMatches, 2);

	vector<DMatch> finalMatches;
	vector<Point2f> srcPoints, dstPoints;

	for (int i = 0; i < initialMatches.size(); ++i)
	{
		if (initialMatches[i][0].distance / initialMatches[i][1].distance <= 0.7) {
			finalMatches.push_back(initialMatches[i][0]);
			srcPoints.push_back(trainKeypoints[initialMatches[i][0].trainIdx].pt);
			dstPoints.push_back(queryKeypoints[initialMatches[i][0].queryIdx].pt);
		}
	}

	vector<uchar> mask;
	Mat H = findHomography(srcPoints, dstPoints, CV_RANSAC, 3.0, mask);

	Mat matchesImage;
	drawMatches(queryImage, queryKeypoints, trainImage, trainKeypoints, finalMatches, matchesImage, Scalar::all(-1), Scalar::all(-1), reinterpret_cast<const vector<char>&>(mask));

	imshow("matches", matchesImage);

	vector<Point2f> srcCorners;
	srcCorners.push_back(Point(0, 0));
	srcCorners.push_back(Point(trainImage.cols, 0));
	srcCorners.push_back(Point(trainImage.cols, trainImage.rows));
	srcCorners.push_back(Point(0, trainImage.rows));

	vector<Point2f> dstConers;
	perspectiveTransform(srcCorners, dstConers, H);

	vector<Point> dstCorners2;
	for (int i = 0; i < dstConers.size(); ++i)
	{
		dstCorners2.push_back(dstConers[i]);
	}

	Mat outputImage;
	cvtColor(queryImage, outputImage, CV_GRAY2BGR);
	fillConvexPoly(outputImage, &dstCorners2[0], dstCorners2.size(), Scalar(0, 255, 0));

	imshow("output", outputImage);

	waitKey();
	return 0;
}*/

int main()
{
	Size boardSize(9, 6);

	VideoCapture capture("left%02d.jpg");

	vector<Mat> frames;
	Mat curFrame;

	vector<vector<Point2f>> imagePoints;

	while (capture.read(curFrame))
	{
		frames.push_back(curFrame.clone());

		vector<Point2f> ptvec;

		bool found = findChessboardCorners(curFrame, boardSize, ptvec);

		if (found)
		{
			Mat gray = curFrame.clone();
			cvtColor(curFrame, curFrame, CV_GRAY2BGR);

			cornerSubPix(gray, ptvec, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

			imagePoints.push_back(ptvec);
			drawChessboardCorners(curFrame, boardSize, ptvec, found);
		}

		imshow("source", curFrame);
	//	waitKey();
	}

	int squareSize = 25;
	vector<Point3f> corners;

	for (int i = 0; i < boardSize.height; ++i)
	{
		for (int j = 0; j < boardSize.width; ++j)
		{
			corners.push_back(Point3f(float(j * squareSize), float(i * squareSize), 0));
		}
	}

	vector<vector<Point3f>> objectPoints;
	for (int i = 0; i < imagePoints.size(); ++i)
	{
		objectPoints.push_back(corners);
	}

	Mat K, distCoeffs;
	vector<Mat> rvecs, tvecs;

	double reprojError = calibrateCamera(objectPoints, imagePoints, Size(frames[0].cols, frames[0].rows), K, distCoeffs, rvecs, tvecs);

	cout << "K: " << K << endl;
	cout << "dist coeffs: " << distCoeffs << endl;
	cout << "reproj error: " << reprojError << endl;

	for (int i = 0; i < frames.size(); ++i)
	{
		Mat undistorted;
		undistort(frames[i], undistorted, K, distCoeffs);
		
		imshow("source", frames[i]);
		imshow("undistorted", undistorted);

		Mat rvec, tvec;
		solvePnPRansac(corners, imagePoints[i], K, distCoeffs, rvec, tvec);

		vector<Point2f> projPoints;
		projectPoints(corners, rvec, tvec, K, distCoeffs, projPoints);

		Mat output;
		cvtColor(frames[i], output, CV_GRAY2BGR);

		for (int j = 0; j < projPoints.size(); ++j)
		{
			circle(output, projPoints[j], 3, Scalar(0, 0, 255));
		}

		imshow("output", output);

		waitKey();
	}

	waitKey();

	return 0;
}

/*
Mat trainImage;
Mat queryImage;

int state = 0;

void onMouse(int type, int x, int y, int, void*) {
	if (state == 0)
		state = 1;

	if (type == CV_EVENT_LBUTTONDOWN) {
		state = 2;
	}
}

int main() {
	VideoCapture capture(0);

	Ptr<Feature2D> detector = ORB::create();

	while (true) {
		if (state == 0) {
			setMouseCallback("train", onMouse);
		}

		if (state < 2) {
			capture >> trainImage;

			imshow("train", trainImage);
		} 
		else 
		{
			capture >> queryImage;

			vector<KeyPoint> trainKeypoints;
			Mat trainDescriptors;
			detector->detectAndCompute(trainImage, noArray(), trainKeypoints, trainDescriptors);

			vector<KeyPoint> queryKeypoints;
			Mat queryDescriptors;
			detector->detectAndCompute(queryImage, noArray(), queryKeypoints, queryDescriptors);

			BFMatcher matcher(NORM_HAMMING);

			vector<vector<DMatch>> initialMatches;
			matcher.knnMatch(queryDescriptors, trainDescriptors, initialMatches, 2);

			vector<DMatch> finalMatches;
			for (int i = 0; i < initialMatches.size(); ++i)
			{
				if (initialMatches[i][0].distance / initialMatches[i][1].distance <= 0.7) {
					finalMatches.push_back(initialMatches[i][0]);
				}
			}

			Mat matchesImage;
			drawMatches(queryImage, queryKeypoints, trainImage, trainKeypoints, finalMatches, matchesImage);

			imshow("train", matchesImage);
		}

		if (waitKey(1) == 27)
			break;
	}

	/*Ptr<Feature2D> detector = BRISK::create();

	vector<KeyPoint> trainKeypoints;
	Mat trainDescriptors;
	detector->detectAndCompute(trainImage, noArray(), trainKeypoints, trainDescriptors);

	vector<KeyPoint> queryKeypoints;
	Mat queryDescriptors;
	detector->detectAndCompute(queryImage, noArray(), queryKeypoints, queryDescriptors);

	BFMatcher matcher(NORM_HAMMING);
	
	vector<vector<DMatch>> initialMatches;
	matcher.knnMatch(queryDescriptors, trainDescriptors, initialMatches, 2);

	vector<DMatch> finalMatches;
	for (int i = 0; i < initialMatches.size(); ++i)
	{
		if (initialMatches[i][0].distance / initialMatches[i][1].distance <= 0.7) {
			finalMatches.push_back(initialMatches[i][0]);
		}
	}

	Mat matchesImage;
	drawMatches(queryImage, queryKeypoints, trainImage, trainKeypoints, finalMatches, matchesImage);

	imshow("matches", matchesImage);
	waitKey();

	return 0;
}

/*Mat src;
Point2f center;
int angle;
int scale = 10;

int currentPoint = 0;
Point2f points[6];

void transform_callback(int, void*) {
	if (currentPoint > 5) {
		vector<Point2f> pointsSRC;
		pointsSRC.push_back(Point2f(points[0].x, points[0].y));
		pointsSRC.push_back(Point2f(points[1].x, points[1].y));
		pointsSRC.push_back(Point2f(points[2].x, points[2].y));

		vector<Point2f> pointsDST;
		pointsDST.push_back(Point2f(points[3].x, points[3].y));
		pointsDST.push_back(Point2f(points[4].x, points[4].y));
		pointsDST.push_back(Point2f(points[5].x, points[5].y));

		Mat rot = getAffineTransform(pointsSRC, pointsDST);

		Mat dst;
		warpAffine(src, dst, rot, src.size());

		imshow("Rotation", dst);
	}
	else {
		imshow("Rotation", src);
	}
}

void onMouse(int type, int x, int y, int, void*) {
	if (type == CV_EVENT_LBUTTONDOWN) {
		if (currentPoint > 5)
			currentPoint = 0;

		points[currentPoint] = Point2f(x, y);
		currentPoint++;

		transform_callback(0, 0);
	}
}

int main() {
	src = imread("Lenna.png");

	//center.x = src.cols / 2;
	//center.y = src.rows / 2;

	imshow("Source", src);

	//createTrackbar("Angle:", "Source", &angle, 360, transform_callback);
	//createTrackbar("Scale:", "Source", &scale, 20, transform_callback);

	setMouseCallback("Source", onMouse);

	transform_callback(0, 0);

	waitKey(0);

	return 0;
}

/*
Mat src_gray;
int thresh = 100;
int maxLevel = 1;
RNG rng(12345);

void thresh_callback(int, void*) {
	Mat binary;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	threshold(src_gray, binary, thresh, 255, THRESH_BINARY);

	findContours(binary.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	Mat drawing = Mat::zeros(binary.size(), CV_8UC3);
	double max = 0;
	int index = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > max) {
			max = area;
			index = i;
		}
	}

	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	drawContours(drawing, contours, index, color, 1, 8, hierarchy, maxLevel);

	imshow("Binary", binary);
	imshow("Contours", drawing);
}

int main() {
	Mat src = imread("Lenna.png");
	cvtColor(src, src_gray, CV_BGR2GRAY);

	namedWindow("Binary", CV_WINDOW_AUTOSIZE);
	createTrackbar("Thresh:", "Binary", &thresh, 255, thresh_callback);
	createTrackbar("MaxLevel:", "Binary", &maxLevel, 100, thresh_callback);

	thresh_callback(0, 0);

	waitKey(0);

	return 0;
}
*/
/*
Mat src;
Point2f center;
int angle;
int scale = 10;

void transform_callback(int, void*) {
	Mat rot = getRotationMatrix2D(center, angle, (double)scale / 10);

	Mat dst;
	warpAffine(src, dst, rot, src.size());

	imshow("Rotation", dst);
}

void onMouse(int type, int x, int y, int, void*) {
	if (type == CV_EVENT_LBUTTONDOWN) {
		center.x = x;
		center.y = y;

		transform_callback(0, 0);
	}
}

int main() {
	src = imread("Lenna.png");

	center.x = src.cols / 2;
	center.y = src.rows / 2;

	imshow("Source", src);

	createTrackbar("Angle:", "Source", &angle, 360, transform_callback);
	createTrackbar("Scale:", "Source", &scale, 20, transform_callback);

	setMouseCallback("Source", onMouse);

	transform_callback(0, 0);

	waitKey(0);

	return 0;
}*/

/*
int main() {
	Mat src = imread("Lenna.png");
	
	vector<Mat> pyramid;
	buildPyramid(src, pyramid, 5);

	for (int i = 0; i < pyramid.size(); i++)
	{
		imshow("Level " + to_string(i), pyramid[i]);
	}

	waitKey(0);

	return 0;
}

/* AULAS ANTERIORES 

void aula01() {
	Mat newImg, new2;
	cvtColor(imread("Lenna.png"), newImg, CV_BGR2GRAY);
	cvtColor(imread("Lenna.png"), new2, CV_RGB2GRAY);

	imshow("image", newImg);
	imshow("image2", new2);
	imshow("image3", newImg + new2);
	imwrite("LennaPB.png", newImg);

	waitKey();
}

void aula02() {
	VideoCapture capture(0);

	while (true) {
		Mat frame, frame2;
		capture >> frame;

		cvtColor(frame, frame, CV_BGR2GRAY);
		flip(frame, frame2, 1);

		for (int i = 0; i < frame2.rows; i++)
		{
			uchar* Mi = frame2.ptr<uchar>(i);

			for (int j = 0; j < frame2.cols; j++)
			{
				Mi[j] = saturate_cast<uchar>(Mi[j] * 2.0);
			}
		}

		imshow("camera", frame);
		imshow("clara", frame2);

		if (waitKey(1) == 27)
			break;
	}
}

// 03-01

void Threshold_Demo(int, void*) {
	Mat dst;
	threshold(src_gray, dst, value, maxValue, type);
	imshow("Threshold", dst);
}

void aula03_01() {
	cvtColor(imread("sudoku.jpg"), src_gray, CV_BGR2GRAY);

	namedWindow("Threshold", CV_WINDOW_AUTOSIZE);

	createTrackbar("Type", "Threshold", &type, 4, Threshold_Demo);
	createTrackbar("Value", "Threshold", &value, 255, Threshold_Demo);
	createTrackbar("Max Value", "Threshold", &maxValue, 255, Threshold_Demo);

	Threshold_Demo(0, 0);

	while (true) {
		if (waitKey(1) == 27) {
			break;
		}
	}
}

// 03-2

int value = 0;
int type = 3;
int maxValue = 255;

int maskSize = 10;
int a = 5;
int filter = 3;

Mat src_gray;

void Blur(int, void*) {
	Mat dst;

	maskSize = maskSize > 0 ? maskSize : 1;

	if (filter == 0)
		blur(src_gray, dst, Size(maskSize, maskSize));
	else if (filter == 1)
		GaussianBlur(src_gray, dst, Size(maskSize * 2 + 1, maskSize * 2 + 1), 0);
	else if (filter == 2)
		medianBlur(src_gray, dst, maskSize * 2 + 1);
	else if (filter == 3) {
		maskSize = maskSize > 15 ? 15 : maskSize;

		Sobel(src_gray, dst, CV_16S, 1, 1, maskSize * 2 + 1);
		convertScaleAbs(dst, dst);
	}

	Mat dst2 = ((float)a / 10 + 1) * src_gray - dst;

	imshow("Blur", dst2);
}

void aula03_02() {
	cvtColor(imread("Lenna.png"), src_gray, CV_BGR2GRAY);

	namedWindow("Blur", CV_WINDOW_AUTOSIZE);

	createTrackbar("Mask", "Blur", &maskSize, 50, Blur);
	createTrackbar("A", "Blur", &a, 30, Blur);
	createTrackbar("Filter", "Blur", &filter, 3, Blur);

	Blur(0, 0);

	while (true) {
		if (waitKey(1) == 27) {
			break;
		}
	}
}

// 04

Mat src_gray;
int thresh = 100;
RNG rng(12345);

void thresh_callback(int, void*) {
	Mat binary;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	threshold(src_gray, binary, thresh, 255, THRESH_BINARY);

	findContours(binary.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	Mat drawing = Mat::zeros(binary.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	drawContours(drawing, contours, i, color, 1, 4 );
	}

	imshow("Binary", binary);
	imshow("Contours", drawing);
}

void aula04_01() {
	Mat src = imread("Lenna.png");
	cvtColor(src, src_gray, CV_BGR2GRAY);

	namedWindow("Binary", CV_WINDOW_AUTOSIZE);
	createTrackbar("Thresh:", "Binary", &thresh, 255, thresh_callback);

	thresh_callback(0, 0);

	waitKey(0);

	return 0;
}
*/