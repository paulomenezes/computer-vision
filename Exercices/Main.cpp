#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

void questao1()
{
	RNG rng(12345);

	Mat image = imread("polygons.png", CV_LOAD_IMAGE_GRAYSCALE);
	GaussianBlur(image, image, Size(3, 3), 2);
	threshold(image, image, 150, 255, CV_THRESH_BINARY);
	Canny(image, image, 50, 200);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	dilate(image, image, Mat());

	findContours(image, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	Mat drawing = Mat::zeros(image.size(), CV_8UC3);

	for (size_t k = 0; k < contours.size(); k += 2) {
		approxPolyDP(contours[k], contours[k], 5, true);

		cout << "quantidade de vertices: " << contours[k].size() << endl;
		for (size_t i = 0; i < contours[k].size(); i++)
			cout << "x " << contours[k][i].x << " y " << contours[k][i].y << endl;

		cout << "\n" << endl;

		Scalar color(rand() & 255, rand() & 255, rand() & 255);
		drawContours(drawing, contours, k, color);
	}

	imshow("drawing", drawing);
}

vector<KeyPoint> keypoints_object, keypoints_scene;

vector<DMatch> final_matches;
Mat image1;
Mat image2;
int offset = 134;

void trackbar(int, void*)
{
	vector<Point2f> src, dst;
	vector<KeyPoint> keypoints_object_src, keypoints_scene_dst;

	for (int i = 0; i < 4; i++)
	{
		src.push_back(keypoints_object[final_matches[offset + i].queryIdx].pt);
		dst.push_back(keypoints_scene[final_matches[offset + i].trainIdx].pt);
	}

	Mat transform = findHomography(src, dst, CV_RANSAC); // getPerspectiveTransform(src, dst);
	Mat image_final;
	warpPerspective(image2, image_final, transform, image2.size());

	Mat matches_image;
	drawMatches(image1, keypoints_object, image_final, keypoints_scene, final_matches, matches_image);
	imshow("Images", matches_image);
}

void questao2() 
{
	image1 = imread("Image1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	image2 = imread("Image2.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	imshow("image1", image1);
	//imshow("image2", image2);

	Ptr<Feature2D> detector = ORB::create();

	Mat descriptors_object, descriptors_scene;
	detector->detectAndCompute(image1, noArray(), keypoints_object, descriptors_object);
	detector->detectAndCompute(image2, noArray(), keypoints_scene, descriptors_scene);

	BFMatcher matcher(NORM_HAMMING);
	vector<vector<DMatch>> initial_matches;
	matcher.knnMatch(descriptors_object, descriptors_scene, initial_matches, 2);

	for (int i = 0; i < initial_matches.size(); i++)
	{
		if (initial_matches[i][0].distance / initial_matches[i][1].distance <= 0.7) {
			final_matches.push_back(initial_matches[i][0]);
		}
	}

	createTrackbar("Offset", "image1", &offset, final_matches.size() - 5, trackbar);

	trackbar(0, 0);
}

int main()
{
	//questao1();
	questao2();

	waitKey();
	return 0;
}